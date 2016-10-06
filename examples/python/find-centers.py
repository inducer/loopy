import numpy as np
import loopy as lp
import pyopencl as cl

cl_ctx = cl.create_some_context(interactive=True)

knl = lp.make_kernel(
    "{[ictr,itgt,idim]: "
    "0<=itgt<ntargets "
    "and 0<=ictr<ncenters "
    "and 0<=idim<ambient_dim}",

    """
    for itgt
        for ictr
            <> dist_sq = sum(idim,
                    (tgt[idim,itgt] - center[idim,ictr])**2)
            <> in_disk = dist_sq < (radius[ictr]*1.05)**2
            <> matches = (
                    (in_disk
                        and qbx_forced_limit == 0)
                    or (in_disk
                            and qbx_forced_limit != 0
                            and qbx_forced_limit * center_side[ictr] > 0)
                    )

            <> post_dist_sq = if(matches, dist_sq, HUGE)
        end
        <> min_dist_sq, <> min_ictr = argmin(ictr, post_dist_sq)

        tgt_to_qbx_center[itgt] = if(min_dist_sq < HUGE, min_ictr, -1)
    end
    """)

knl = lp.fix_parameters(knl, ambient_dim=2)
knl = lp.add_and_infer_dtypes(knl, {
        "tgt,center,radius,HUGE": np.float32,
        "center_side,qbx_forced_limit": np.int32,
        })

lp.auto_test_vs_ref(knl, cl_ctx, knl, parameters={
        "HUGE": 1e20, "ncenters": 200, "ntargets": 300,
        "qbx_forced_limit": 1})

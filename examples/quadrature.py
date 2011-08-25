
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp




def make_well_conditioned_dev_matrix(queue, shape, dtype=np.float32, 
        order="C", ran_factor=1, id_factor=5, inc_factor=0, od=0):
    if isinstance(shape, int):
        shape = (shape, shape)
    l = max(shape)
    eye_ish = id_factor*np.eye(l, k=od)
    if inc_factor:
        eye_ish[np.arange(l), np.arange(l)] = inc_factor*np.arange(l)
    ary = np.asarray(
        ran_factor*np.random.randn(*shape)
        + eye_ish[:shape[0], :shape[1]],
        dtype=dtype, order=order)

    return cl_array.to_device(queue, ary)




def build_mass_mat_maker(ctx_factory=cl.create_some_context):
    dtype = np.float32
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    Nb = 3
    Nv = 3
    Nq = 3*3

    Nc = 1600
    from pymbolic import var
    m, w, det_j, phi, c, i, j, q = [var(s) for s in "m w det_j phi c i j q".split()]

    knl = lp.LoopKernel(ctx.devices[0],
            "[ncells] -> {[c,i,j,q]: 0<=c<ncells and 0 <= i < %(Nv)s "
            "and 0<=j<%(Nb)s and 0<=q<%(Nq)s}" % dict(
                Nv=Nv, Nb=Nb, Nq=Nq),
            [
                (m[c, i, j], w[q]*det_j[c]*phi[i,q]*phi[j,q])
                ],
            [
                lp.ArrayArg("m", dtype, shape=(Nc, Nv, Nb)),
                lp.ArrayArg("w", dtype, shape=(Nq,)),
                lp.ArrayArg("det_j", dtype, shape=(Nc,)),
                lp.ArrayArg("phi", dtype, shape=(Nv, Nq,)),
                lp.ScalarArg("ncells", np.int32, approximately=1000),
                ],
            name="mass_mat",
            iname_to_tag=dict(i="l.0", j="l.1")
            )
    knl = lp.split_dimension(knl, "c", 8, outer_tag="g.0", inner_tag="l.2")
    knl = lp.add_prefetch(knl, "det_j", ["c_inner"])

    # fix reg prefetch
    # fix redundant slab generation

    # FIXME
    #knl = lp.split_dimension(knl, "c", 8, inner_tag="l.2")
    #knl = lp.split_dimension(knl, "c_outer", 8, outer_tag="g.0")

    #ilp = 4
    #knl = lp.split_dimension(knl, "i", 2, outer_tag="g.0", inner_tag="l.1")
    #j_inner_split = 16
    #knl = lp.split_dimension(knl, "j", ilp*j_inner_split, outer_tag="g.1")
    #knl = lp.split_dimension(knl, "j_inner", j_inner_split, outer_tag="ilp", inner_tag="l.0")
    #knl = lp.split_dimension(knl, "k", 2)

    #knl = lp.add_prefetch(knl, 'a', ["i_inner", "k_inner"])
    #knl = lp.add_prefetch(knl, 'b', ["j_inner_outer", "j_inner_inner", "k_inner"])
    #assert knl.get_problems({})[0] <= 2

    kernel_gen = (lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl))

    if False:
        a = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order,
                ran_factor=1, id_factor=5)
        b = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order,
                ran_factor=1, id_factor=5, inc_factor=0)
        c = cl_array.empty_like(a)
        a_img = cl.image_from_array(ctx, a.get(), 1)
        b_img = cl.image_from_array(ctx, b.get(), 1)

    def launcher(kernel, gsize, lsize, check):
        1/0
        evt = kernel(queue, gsize(), lsize(), a_img, b_img, c.data,
                g_times_l=True)

        return evt

    from pyopencl.characterize import get_fast_inaccurate_build_options
    lp.drive_timing_run(kernel_gen, queue, launcher, flop_count=0,
            options=get_fast_inaccurate_build_options(ctx.devices[0]))




if __name__ == "__main__":
    build_mass_mat_maker()

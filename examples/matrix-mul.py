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




def image_matrix_mul_ilp(ctx_factory=cl.create_some_context):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"
    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    n = 16*10
    from pymbolic import var
    a, b, c, i, j, k, n_sym = [var(s) for s in "abcijkn"]

    knl = lp.LoopKernel(ctx.devices[0],
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                (c[i, j], a[i, k]*b[k, j])
                ],
            [
                lp.ImageArg("a", dtype, 2),
                lp.ImageArg("b", dtype, 2),
                lp.GlobalArg("c", dtype, shape=(n, n), order=order),
                ],
            name="matmul")

    ilp = 4
    knl = lp.split_dimension(knl, "i", 2, outer_tag="g.0", inner_tag="l.1")
    j_inner_split = 16
    knl = lp.split_dimension(knl, "j", ilp*j_inner_split, outer_tag="g.1")
    knl = lp.split_dimension(knl, "j_inner", j_inner_split, outer_tag="ilp", inner_tag="l.0")
    knl = lp.split_dimension(knl, "k", 2)

    knl = lp.add_prefetch(knl, 'a', ["i_inner", "k_inner"])
    knl = lp.add_prefetch(knl, 'b', ["j_inner_outer", "j_inner_inner", "k_inner"])
    assert knl.get_problems({})[0] <= 2

    kernel_gen = (lp.insert_register_prefetches(knl)
            for knl in lp.generate_loop_schedules(knl))

    a = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order,
            ran_factor=1, id_factor=5)
    b = make_well_conditioned_dev_matrix(queue, n, dtype=dtype, order=order,
            ran_factor=1, id_factor=5, inc_factor=0)
    c = cl_array.empty_like(a)
    a_img = cl.image_from_array(ctx, a.get(), 1)
    b_img = cl.image_from_array(ctx, b.get(), 1)

    def launcher(kernel, gsize, lsize, check):
        evt = kernel(queue, gsize(), lsize(), a_img, b_img, c.data,
                g_times_l=True)

        return evt

    from pyopencl.characterize import get_fast_inaccurate_build_options
    lp.drive_timing_run(kernel_gen, queue, launcher, 2*n**3,
            options=get_fast_inaccurate_build_options(ctx.devices[0]))




if __name__ == "__main__":
    image_matrix_mul_ilp()

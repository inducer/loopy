#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))

__kernel void __attribute__ ((reqd_work_group_size(128, 1, 1)))
  loopy_kernel(__global float *restrict out, __global float const *restrict a, int const n)
{

    if ((-1 + -128 * gid(0) + -1 * lid(0) + n) >= 0)
          out[lid(0) + gid(0) * 128] = 2.0f * a[lid(0) + gid(0) * 128];
}

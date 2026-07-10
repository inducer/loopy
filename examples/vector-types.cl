#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))
#define int_floor_div_pos_b(a,b) (                 ( (a) - ( ((a)<0) ? ((b)-1) : 0 )  ) / (b)                 )

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global float4 const *__restrict__ a, int const n, __global float4 *__restrict__ out)
{
  /* bulk slab for 'i_outer' */
  for (int i_outer = 0; i_outer <= -2 + int_floor_div_pos_b(3 + n, 4); ++i_outer)
    out[i_outer] = 2.0f * a[i_outer];
  /* final slab for 'i_outer' */
  {
    int const i_outer = -1 + n + -1 * int_floor_div_pos_b(3 * n, 4);

    if (-1 + n >= 0)
    {
      if (-1 + -4 * i_outer + n >= 0)
        out[i_outer].s0 = 2.0f * a[i_outer].s0;
      if (-1 + -4 * i_outer + -1 + n >= 0)
        out[i_outer].s1 = 2.0f * a[i_outer].s1;
      if (-1 + -4 * i_outer + -1 * 2 + n >= 0)
        out[i_outer].s2 = 2.0f * a[i_outer].s2;
      if (-1 + -4 * i_outer + -1 * 3 + n >= 0)
        out[i_outer].s3 = 2.0f * a[i_outer].s3;
    }
  }
}

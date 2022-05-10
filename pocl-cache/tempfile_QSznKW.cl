#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))
#if __OPENCL_C_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global double *__restrict__ y1, __global double *__restrict__ y2, __global float *__restrict__ y3)
{
  double tmp1;
  float tmp2;

  tmp1 = 3.1416;
  tmp2 = 0.0f;
  y1[0] = (tmp1 != 0 ? 1729.0 : 1.414);
  y2[0] = (2.7183 != 0 ? 42.0 : 13.0);
  y3[0] = (tmp2 != 0 ? 127.0f : 128.0f);
}

__constant int pyopencl_defeat_cache_382f60a581b24f8f9235ae565743c623 = 0;
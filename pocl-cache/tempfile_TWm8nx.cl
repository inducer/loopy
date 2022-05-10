//CL//
        

        #define PYOPENCL_ELWISE_CONTINUE continue

        __kernel void scalar_comparison_kernel(__global char *out__base, long out__offset, __global float *a__base, long a__offset, float b, long n)
        {
          int lid = get_local_id(0);
          int gsize = get_global_size(0);
          int work_group_start = get_local_size(0)*get_group_id(0);
          long i;

          __global char *out = (__global char *) ((__global char *) out__base + out__offset);
__global float *a = (__global float *) ((__global char *) a__base + a__offset);;
          //CL//
          for (i = work_group_start + lid; i < n; i += gsize)
          {
            out[i] = a[i] == b;
          }
          
          ;
        }
        

__constant int pyopencl_defeat_cache_ebc0099bd26d433e925c36767245cf6b = 0;
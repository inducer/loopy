import numpy as np
import loopy as lp
import pyopencl as cl
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
knl = lp.make_kernel(
    "{[i]: 0<=i<n}",
    """
    y[i] = a[i] + 2*b[i] + 3*c[n-1-i]
    """)
a = b = c = np.arange(15)
knl = lp.add_dtypes(knl, {"a,b,c": np.int64})
# print(knl)
print(lp.generate_code_v2(knl).device_code())
cuda_knl = knl.copy(target=lp.CudaTarget())
print(lp.generate_code_v2(cuda_knl).device_code())
cuda_knl(a=a, b=b, c=c)
lp.generate_code_v2(cuda_knl).host_code()
print(lp.generate_code_v2(knl).host_code())

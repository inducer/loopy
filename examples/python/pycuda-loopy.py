import numpy as np
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

import pycuda as cuda

a = np.arange(15, dtype=np.float32)

knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]")

knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")
cuda_knl = knl.copy(target=lp.CudaTarget())
evt, (out,) = cuda_knl(a=a)

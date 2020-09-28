import numpy as np
import loopy as lp

knl = lp.make_kernel(
        "{ [i,k]: 0<=i<n and 0<=k<3 }",
        """
        for i, k
            ... gbarrier
            c[k,i] = a[k, i + 1]
            ... gbarrier
            out[k,i] = c[k,i]
        end
        """, seq_dependencies=True)

# transform
knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")
knl = lp.add_and_infer_dtypes(knl,
        {"a": np.float32, "c": np.float32, "out": np.float32, "n": np.int32})

# schedule
from loopy.preprocess import preprocess_kernel
knl = preprocess_kernel(knl)

from loopy.schedule import get_one_scheduled_kernel
knl = get_one_scheduled_kernel(knl)

# map schedule onto host or device
print(knl)

cgr = lp.generate_code_v2(knl)

print(cgr.device_code())
print(cgr.host_code())

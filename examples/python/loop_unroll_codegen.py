import numpy as np

import pyopencl as cl
import pyopencl.array

import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


# setup
# -----
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 15 * 10**6
a = cl.array.arange(queue, n, dtype=np.float32)

# create
# ------
knl = lp.make_kernel(
        "{ [i]: 0<= i <8}",
        "out[i] = a if i == 0 else (b if i == 1 else c)")

knl = lp.tag_inames(knl, {"i": "vec"})
from loopy.kernel.array import VectorArrayDimTag


try:
    orig_knl = knl
    knl = lp.tag_array_axes(knl, "out", [VectorArrayDimTag()])
    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32,
                                        "b": np.float32,
                                        "c": np.float32})

    dev_code = lp.generate_code_v2(knl).device_code()
    print(dev_code)

except Exception as err:
    print(err)

print("No Vector Array Tag.")
knl = orig_knl
knl = lp.make_kernel(
        "{ [i]: 0<= i <8}",
        "out[i] = a if i == 0 else (b if i == 1 else c)")

knl = lp.tag_inames(knl, {"i": "ilp.unr"})
knl = lp.add_and_infer_dtypes(knl, {"a": np.float32, "b": np.float32, "c": np.float32})
dev_code = lp.generate_code_v2(knl).device_code()
print(dev_code)

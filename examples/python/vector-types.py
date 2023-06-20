import numpy as np
import pyopencl as cl
import pyopencl.array

import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 15 * 10**6
a = cl.array.arange(queue, n, dtype=np.float32)

knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]")

knl = lp.set_options(knl, write_code=True)
knl = lp.split_iname(knl, "i", 4, slabs=(0, 1), inner_tag="vec")
knl = lp.split_array_axis(knl, "a,out", axis_nr=0, count=4)
knl = lp.tag_array_axes(knl, "a,out", "C,vec")

knl(queue, a=a.reshape(-1, 4), n=n)

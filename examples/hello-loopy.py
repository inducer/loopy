import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array

# -----------------------------------------------------------------------------
# setup
# -----------------------------------------------------------------------------
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 15 * 10**6
a = cl.array.arange(queue, n, dtype=np.float32)

# -----------------------------------------------------------------------------
# generation (loopy bits start here)
# -----------------------------------------------------------------------------
knl = lp.make_kernel(
        ctx.devices[0],
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]")

# -----------------------------------------------------------------------------
# transformation
# -----------------------------------------------------------------------------
knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

# -----------------------------------------------------------------------------
# execution
# -----------------------------------------------------------------------------
cknl = lp.CompiledKernel(ctx, knl)
evt, (out,) = cknl(queue, a=a, n=n)

print cknl.get_highlighted_code({"a": np.float32})

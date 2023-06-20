# SETUPBEGIN
import numpy as np
import pyopencl as cl

import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

knl = lp.make_kernel(
    "{[i, j]: 0<=i<n and 0<=j<n}",
    "c[i, j] = a[i]*b[j]",
    assumptions="n >= 16")

a = np.arange(200, dtype=np.float32)
b = np.arange(200, dtype=np.float32)

knl = lp.set_options(knl, write_code=True)
evt, (c,) = knl(queue, a=a, b=b)
# SETUPEND

orig_knl = knl

# SPLITBEGIN
knl = lp.split_iname(knl, "i", 16,
        outer_tag="g.0", inner_tag="l.0")
knl = lp.split_iname(knl, "j", 16,
        outer_tag="g.1", inner_tag="l.1")
# SPLITEND

knl = lp.set_options(knl, write_code=True)
evt, (c,) = knl(queue, a=a, b=b)

split_knl = knl

# PREFETCH1BEGIN
knl = lp.add_prefetch(knl, "a",
        fetch_outer_inames="i_outer, i_inner, j_outer, j_inner")
knl = lp.add_prefetch(knl, "b",
        fetch_outer_inames="i_outer, i_inner, j_outer, j_inner")
# PREFETCH1END

knl = lp.set_options(knl, write_code=True)
evt, (c,) = knl(queue, a=a, b=b)

knl = split_knl

# PREFETCH2BEGIN
knl = lp.add_prefetch(knl, "a", ["i_inner"],
        fetch_outer_inames="i_outer, j_outer, j_inner",
        temporary_address_space=lp.AddressSpace.LOCAL,
        default_tag="l.0")
knl = lp.add_prefetch(knl, "b", ["j_inner"],
        fetch_outer_inames="i_outer, j_outer, j_inner",
        temporary_address_space=lp.AddressSpace.LOCAL,
        default_tag="l.0")
# PREFETCH2END

knl = lp.set_options(knl, write_code=True)
evt, (c,) = knl(queue, a=a, b=b)

knl = orig_knl

# PREFETCH3BEGIN
knl = lp.split_iname(knl, "i", 256,
        outer_tag="g.0", slabs=(0, 1))
knl = lp.split_iname(knl, "j", 256,
        outer_tag="g.1", slabs=(0, 1))

knl = lp.add_prefetch(knl, "a", ["i_inner"],
        fetch_outer_inames="i_outer, j_outer", default_tag=None)
knl = lp.add_prefetch(knl, "b", ["j_inner"],
        fetch_outer_inames="i_outer, j_outer", default_tag=None)

knl = lp.split_iname(knl, "i_inner", 16,
        inner_tag="l.0")
knl = lp.split_iname(knl, "j_inner", 16,
        inner_tag="l.1")

knl = lp.split_iname(knl, "b_dim_0", 16,
        outer_tag="l.1", inner_tag="l.0")
knl = lp.split_iname(knl, "a_dim_0", 16,
        outer_tag="l.1", inner_tag="l.0")
# PREFETCH3END

knl = lp.set_options(knl, write_code=True)
evt, (c,) = knl(queue, a=a, b=b)

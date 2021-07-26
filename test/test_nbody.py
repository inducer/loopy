__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np
import loopy as lp
import pyopencl as cl  # noqa

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import logging
logger = logging.getLogger(__name__)


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


def test_nbody(ctx_factory):
    logging.basicConfig(level=logging.INFO)

    dtype = np.float32
    ctx = ctx_factory()

    knl = lp.make_kernel(
            "[N] -> {[i,j,k]: 0<=i,j<N and 0<=k<3 }",
            [
                "axdist(k) := x[i,k]-x[j,k]",
                "invdist := rsqrt(sum(k, axdist(k)**2))",
                "pot[i] = sum(j, if(i != j, invdist, 0))",
            ], [
                lp.GlobalArg("x", dtype, shape="N,3", order="C"),
                lp.GlobalArg("pot", dtype, shape="N", order="C"),
                lp.ValueArg("N", np.int32),
            ], name="nbody", assumptions="N>=1")

    seq_knl = knl

    def variant_1(knl):
        knl = lp.split_iname(knl, "i", 256,
                outer_tag="g.0", inner_tag="l.0",
                slabs=(0, 1))
        knl = lp.split_iname(knl, "j", 256, slabs=(0, 1))
        return knl

    def variant_cpu(knl):
        knl = lp.expand_subst(knl)
        knl = lp.split_iname(knl, "i", 1024,
                outer_tag="g.0", slabs=(0, 1))
        knl = lp.add_prefetch(knl, "x[i,k]", ["k"], default_tag=None)
        return knl

    def variant_gpu(knl):
        knl = lp.expand_subst(knl)
        knl = lp.split_iname(knl, "i", 256,
                outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_iname(knl, "j", 256)
        knl = lp.add_prefetch(knl, "x[j,k]", ["j_inner", "k"],
                ["x_fetch_j", "x_fetch_k"],
                fetch_outer_inames="i_outer, j_outer", default_tag=None)
        knl = lp.tag_inames(knl, dict(x_fetch_k="unr", x_fetch_j="l.0"))
        knl = lp.add_prefetch(knl, "x[i,k]", ["k"], default_tag=None)
        knl = lp.prioritize_loops(knl, ["j_outer", "j_inner"])
        return knl

    n = 3000

    for variant in [
            #variant_1,
            #variant_cpu,
            variant_gpu
            ]:
        variant_knl = variant(knl)

        lp.auto_test_vs_ref(seq_knl, ctx, variant_knl,
                op_count=[n**2*1e-6], op_label=["M particle pairs"],
                parameters={"N": n})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

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
import pyopencl as cl  # noqa
import loopy as lp

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


def test_tim2d(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    n = 8

    from pymbolic import var
    K_sym = var("K")  # noqa

    field_shape = (K_sym, n, n)

    # K - run-time symbolic
    knl = lp.make_kernel(
            "{[i,j,e,m,o,o2]: 0<=i,j,m,o,o2<n and 0<=e<K}",
            [
                "ur(a,b) := simul_reduce(sum, o, D[a,o]*u[e,o,b])",
                "us(a,b) := simul_reduce(sum, o2, D[b,o2]*u[e,a,o2])",

                #"Gu(mat_entry,a,b) := G[mat_entry,e,m,j]*ur(m,j)",

                "Gux(a,b) := G$x[0,e,a,b]*ur(a,b)+G$x[1,e,a,b]*us(a,b)",
                "Guy(a,b) := G$y[1,e,a,b]*ur(a,b)+G$y[2,e,a,b]*us(a,b)",
                "lap[e,i,j]  = "
                "  simul_reduce(sum, m, D[m,i]*Gux(m,j))"
                "+ simul_reduce(sum, m, D[m,j]*Guy(i,m))"

            ],
            [
                lp.GlobalArg("u", dtype, shape=field_shape, order=order),
                lp.GlobalArg("lap", dtype, shape=field_shape, order=order),
                lp.GlobalArg("G", dtype, shape=(3,)+field_shape, order=order),
                # lp.ConstantArrayArg("D", dtype, shape=(n, n), order=order),
                lp.GlobalArg("D", dtype, shape=(n, n), order=order),
                # lp.ImageArg("D", dtype, shape=(n, n)),
                lp.ValueArg("K", np.int32, approximately=1000),
                ],
            name="semlap2D", assumptions="K>=1")

    knl = lp.fix_parameters(knl, n=n)
    # knl = lp.duplicate_inames(knl, "o", within="id:ur")
    # knl = lp.duplicate_inames(knl, "o", within="id:us")

    seq_knl = knl

    def variant_orig(knl):
        knl = lp.tag_inames(knl, dict(i="l.0", j="l.1", e="g.0"))

        knl = lp.add_prefetch(knl, "D[:,:]", fetch_outer_inames="e",
                default_tag="l.auto")
        knl = lp.add_prefetch(knl, "u[e, :, :]", default_tag="l.auto")

        knl = lp.precompute(knl, "ur(m,j)", ["m", "j"], default_tag="l.auto")
        knl = lp.precompute(knl, "us(i,m)", ["i", "m"], default_tag="l.auto")
        # TODO this adds `a` and `b` to domains, which leads to unused inames

        knl = lp.precompute(knl, "Gux(m,j)", ["m", "j"], default_tag="l.auto")
        knl = lp.precompute(knl, "Guy(i,m)", ["i", "m"], default_tag="l.auto")

        knl = lp.add_prefetch(knl, "G$x[:,e,:,:]", default_tag="l.auto")
        knl = lp.add_prefetch(knl, "G$y[:,e,:,:]", default_tag="l.auto")

        knl = lp.tag_inames(knl, dict(o="unr"))
        knl = lp.tag_inames(knl, dict(m="unr"))

        knl = lp.set_instruction_priority(knl, "id:D_fetch", 5)
        print(knl)

        return knl

    for variant in [variant_orig]:
        K = 1000  # noqa
        lp.auto_test_vs_ref(seq_knl, ctx, variant(knl),
                op_count=[K*(n*n*n*2*2 + n*n*2*3 + n**3 * 2*2)/1e9],
                op_label=["GFlops"],
                parameters={"K": K})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

from __future__ import division

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
import pyopencl as cl
import loopy as lp

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests




def cannot_schedule_test_tim3d_slab(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    Nq = 8

    knl = lp.make_kernel(ctx.devices[0],
            "[E] -> {[i,j, k, o,m, e]: 0<=i,j,k,o,m < Nq and 0<=e<E }",
            """
            ur(a,b,c) := sum(o, D[a,o]*u[e,o,b,c])
            us(a,b,c) := sum(o, D[b,o]*u[e,a,o,c])
            ut(a,b,c) := sum(o, D[c,o]*u[e,a,b,o])

            Gur(a,b,c) := G[0,e,a,b,c]*ur(a,b,c)+G[1,e,a,b,c]*us(a,b,c)+G[2,e,a,b,c]*ut(a,b,c)
            Gus(a,b,c) := G[1,e,a,b,c]*ur(a,b,c)+G[3,e,a,b,c]*us(a,b,c)+G[4,e,a,b,c]*ut(a,b,c)
            Gut(a,b,c) := G[2,e,a,b,c]*ur(a,b,c)+G[4,e,a,b,c]*us(a,b,c)+G[5,e,a,b,c]*ut(a,b,c)

            lapr(a,b,c):= sum(m, D[m,a]*Gur(m,b,c))
            laps(a,b,c):= sum(m, D[m,b]*Gus(a,m,c))
            lapt(a,b,c):= sum(m, D[m,c]*Gut(a,b,m))

            lap[e,i,j,k] = lapr(i,j,k) + laps(i,j,k) + lapt(i,j,k)
            """,
            [
            lp.GlobalArg("u,lap", dtype, shape="E,Nq,Nq,Nq", order=order),
            lp.GlobalArg("G", dtype, shape="6,E,Nq,Nq,Nq", order=order),
            # lp.ConstantArrayArg("D", dtype, shape="Nq,Nq", order=order),
            lp.GlobalArg("D", dtype, shape="Nq, Nq", order=order),
            # lp.ImageArg("D", dtype, shape="Nq, Nq"),
            lp.ValueArg("E", np.int32, approximately=1000),
            ],
            name="semdiff3D", assumptions="E>=1",
            defines={"Nq": Nq})



    def duplicate_os(knl):
        for derivative in "rst":
            knl = lp.duplicate_inames(
                knl, "o",
                within="... < lap"+derivative, suffix="_"+derivative)
        return knl

    def variant_orig(knl):
        # NOTE: Removing this makes the thing unschedulable
        #knl = lp.tag_inames(knl, dict(e="g.0", i="l.0", j="l.1"), )

        knl = lp.precompute(knl, "ur", ["i", "j"], within="... < lapr")
        knl = lp.precompute(knl, "us", ["i", "j"], within="... < lapr")
        knl = lp.precompute(knl, "ut", ["i", "j"], within="... < lapr")

        # prefetch the derivative matrix
        knl = lp.add_prefetch(knl, "D[:,:]")

        knl = duplicate_os(knl)

        print(knl)

        return knl

    seq_knl = duplicate_os(knl)

    #print lp.preprocess_kernel(knl)
    #1/0

    for variant in [variant_orig]:
        kernel_gen = lp.generate_loop_schedules(variant(knl), loop_priority=["e", "i", "j"])
        kernel_gen = lp.check_kernels(kernel_gen, dict(E=1000))

        E = 1000
        lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
                op_count=[-666],
                op_label=["GFlops"],
                parameters={"E": E})




def test_tim3d_slab(ctx_factory):
    dtype = np.float32
    ctx = ctx_factory()
    order = "C"

    Nq = 8

    knl = lp.make_kernel(ctx.devices[0],
            "[E] -> {[i,j, k, o,m, e]: 0<=i,j,k,o,m < Nq and 0<=e<E }",
            """
            ur(a,b,c) := sum(o, D[a,o]*u[e,o,b,c])
            us(a,b,c) := sum(o, D[b,o]*u[e,a,o,c])
            ut(a,b,c) := sum(o, D[c,o]*u[e,a,b,o])

            Gur(a,b,c) := G[0,e,a,b,c]*ur(a,b,c)+G[1,e,a,b,c]*us(a,b,c)+G[2,e,a,b,c]*ut(a,b,c)
            Gus(a,b,c) := G[1,e,a,b,c]*ur(a,b,c)+G[3,e,a,b,c]*us(a,b,c)+G[4,e,a,b,c]*ut(a,b,c)
            #Gut(a,b,c) := G[2,e,a,b,c]*ur(a,b,c)+G[4,e,a,b,c]*us(a,b,c)+G[5,e,a,b,c]*ut(a,b,c)

            Gut(a,b,c) := G[5,e,a,b,c]*ut(a,b,c)

            lapr(a,b,c):= sum(m, D[m,a]*Gur(m,b,c))
            laps(a,b,c):= sum(m, D[m,b]*Gus(a,m,c))
            lapt(a,b,c):= sum(m, D[m,c]*Gut(a,b,m))

            part_r := lapr(i,j,k)
            part_s := laps(i,j,k)
            part_t := lapt(i,j,k)

            lap[e,i,j,k] = part_t #part_r + part_s #
            """,
            [
            lp.GlobalArg("u,lap", dtype, shape="E,Nq,Nq,Nq", order=order),
            lp.GlobalArg("G", dtype, shape="6,E,Nq,Nq,Nq", order=order),
            # lp.ConstantArrayArg("D", dtype, shape="Nq,Nq", order=order),
            lp.GlobalArg("D", dtype, shape="Nq, Nq", order=order),
            # lp.ImageArg("D", dtype, shape="Nq, Nq"),
            lp.ValueArg("E", np.int32, approximately=1000),
            ],
            name="semdiff3D", assumptions="E>=1",
            defines={"Nq": Nq})



    def duplicate_os(knl):
        for derivative in "rst":
            knl = lp.duplicate_inames(
                knl, "o",
                within="... < lap"+derivative, suffix="_"+derivative)
        return knl

    def variant_orig(knl):
        # NOTE: Removing this makes the thing unschedulable
        knl = lp.tag_inames(knl, dict(e="g.0", i="l.0", j="l.1"), )

        if 0:
            for derivative in "rst":
                for iname in "ij":
                    knl = lp.duplicate_inames(
                        knl, iname, within="part_%s" % derivative, suffix="_"+derivative)

        knl = lp.duplicate_inames(knl, "k", within="part_t", suffix="_t", tags=dict(k="ilp"))
        knl = lp.duplicate_inames(knl, "o", within="... < Gut", suffix="_t")

        knl = lp.link_inames(knl, "i", tag="l.0", new_iname="p0")
        knl = lp.link_inames(knl, "j", tag="l.1", new_iname="p1")

        knl = lp.precompute(knl, "Gut", ["p0", "p1"])
        #knl = lp.precompute(knl, "Gur", ["m", "p1"])
        #knl = lp.precompute(knl, "Gus", ["p0", "m"])

        #knl = lp.precompute(knl, "us", ["i", "j"], within="... < lapr")
        #knl = lp.precompute(knl, "ut", ["i", "j"], within="... < lapr")

        knl = lp.precompute(knl, "lapt", ["", "j"])

        # prefetch the derivative matrix
        knl = lp.add_prefetch(knl, "D[:,:]")

        print(knl)

        return knl

    #seq_knl = duplicate_os(knl)
    seq_knl = knl

    #print lp.preprocess_kernel(knl)
    #1/0

    for variant in [variant_orig]:
        kernel_gen = lp.generate_loop_schedules(variant(knl))
        kernel_gen = lp.check_kernels(kernel_gen, dict(E=1000))

        E = 1000
        lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
                op_count=[-666],
                op_label=["GFlops"],
                parameters={"E": E})

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

from __future__ import division

import numpy as np
import loopy as lp
import pyopencl as cl

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

__all__ = ["pytest_generate_tests",
    "cl" # 'cl.create_some_context'
    ]




def test_owed_barriers(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            "{[i]: 0<=i<100}",
            [
                "[i:l.0] <float32> z[i] = a[i]"
                ],
            [lp.GlobalArg("a", np.float32, shape=(100,))]
            )

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen)

    for gen_knl in kernel_gen:
        compiled = lp.CompiledKernel(ctx, gen_knl)
        print compiled.code




def test_wg_too_small(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            "{[i]: 0<=i<100}",
            [
                "[i:l.0] <float32> z[i] = a[i]"
                ],
            [lp.GlobalArg("a", np.float32, shape=(100,))],
            local_sizes={0: 16})

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen)

    for gen_knl in kernel_gen:
        try:
            lp.CompiledKernel(ctx, gen_knl)
        except RuntimeError, e:
            assert "implemented and desired" in str(e)
            pass # expected!
        else:
            assert False # expecting an error




def test_multi_cse(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            "{[i]: 0<=i<100}",
            [
                "[i] <float32> z[i] = a[i] + a[i]**2"
                ],
            [lp.GlobalArg("a", np.float32, shape=(100,))],
            local_sizes={0: 16})

    knl = lp.split_dimension(knl, "i", 16, inner_tag="l.0")
    knl = lp.add_prefetch(knl, "a", [])

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen)

    for gen_knl in kernel_gen:
        compiled = lp.CompiledKernel(ctx, gen_knl)
        print compiled.code





def test_stencil(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j]: 0<= i,j < 32}",
            [
                "[i] <float32> z[i,j] = -2*a[i,j]"
                    " + a[i,j-1]"
                    " + a[i,j+1]"
                    " + a[i-1,j]"
                    " + a[i+1,j]"
                ],
            [
                lp.GlobalArg("a", np.float32, shape=(32,32,))
                ])


    def variant_1(knl):
        knl = lp.add_prefetch(knl, "a", [0, 1])
        return knl

    def variant_2(knl):
        knl = lp.split_dimension(knl, "i", 16, outer_tag="g.1", inner_tag="l.1")
        knl = lp.split_dimension(knl, "j", 16, outer_tag="g.0", inner_tag="l.0")
        knl = lp.add_prefetch(knl, "a", ["i_inner", "j_inner"])
        return knl

    #for variant in [variant_1, variant_2]:
    for variant in [variant_2]:
        kernel_gen = lp.generate_loop_schedules(variant(knl),
                loop_priority=["i_outer", "i_inner_0", "j_0"])
        kernel_gen = lp.check_kernels(kernel_gen)

        for knl in kernel_gen:
            print lp.generate_code(knl)




def test_eq_constraint(ctx_factory):
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j]: 0<= i,j < 32}",
            [
                "a[i] = b[i]"
                ],
            [
                lp.GlobalArg("a", np.float32, shape=(1000,)),
                lp.GlobalArg("b", np.float32, shape=(1000,))
                ])

    knl = lp.split_dimension(knl, "i", 16, outer_tag="g.0")
    knl = lp.split_dimension(knl, "i_inner", 16, outer_tag=None, inner_tag="l.0")

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen)

    for knl in kernel_gen:
        print lp.generate_code(knl)




def test_argmax(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    order = "C"

    n = 10000

    knl = lp.make_kernel(ctx.devices[0],
            "{[i]: 0<=i<%d}" % n,
            [
                "<> result = argmax(i, fabs(a[i]))",
                "max_idx = result.index",
                "max_val = result.value",
                ],
            [
                lp.GlobalArg("a", dtype, shape=(n,), order=order),
                lp.GlobalArg("max_idx", np.int32, shape=(), order=order),
                lp.GlobalArg("max_val", dtype, shape=(), order=order),
                ])

    a = np.random.randn(10000).astype(dtype)
    cknl = lp.CompiledKernel(ctx, knl)
    evt, (max_idx, max_val) = cknl(queue, a=a, out_host=True)
    assert max_val == np.max(np.abs(a))
    assert max_idx == np.where(np.abs(a)==max_val)[-1]




def make_random_value():
    from random import randrange, uniform
    v = randrange(3)
    if v == 0:
        while True:
            z = randrange(-1000, 1000)
            if z:
                return z

    elif v == 1:
        return uniform(-10, 10)
    else:
        return uniform(-10, 10) + 1j*uniform(-10, 10)




def make_random_expression(var_values, size):
    from random import randrange
    import pymbolic.primitives as p
    v = randrange(1500)
    size[0] += 1
    if v < 500 and size[0] < 40:
        term_count = randrange(2, 5)
        if randrange(2) < 1:
            cls = p.Sum
        else:
            cls = p.Product
        return cls(tuple(
            make_random_expression(var_values, size)
            for i in range(term_count)))
    elif v < 750:
        return make_random_value()
    elif v < 1000:
        var_name = "var_%d" % len(var_values)
        assert var_name not in var_values
        var_values[var_name] = make_random_value()
        return p.Variable(var_name)
    elif v < 1250:
        return make_random_expression(var_values, size) - make_random_expression(var_values, size)
    elif v < 1500:
        return make_random_expression(var_values, size) / make_random_expression(var_values, size)


def generate_random_fuzz_examples(count):
    for i in xrange(count):
        size = [0]
        var_values = {}
        expr = make_random_expression(var_values, size)
        yield expr, var_values

def test_fuzz_code_generator(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    #from expr_fuzz import get_fuzz_examples
    for expr, var_values in generate_random_fuzz_examples(20):
    #for expr, var_values in get_fuzz_examples():
        from pymbolic import evaluate
        true_value = evaluate(expr, var_values)

        def get_dtype(x):
            if isinstance(x, complex):
                return np.complex128
            else:
                return np.float64

        knl = lp.make_kernel(ctx.devices[0], "{ : }",
                [lp.Instruction(None, "value", expr)],
                [lp.GlobalArg("value", np.complex128, shape=())]
                + [
                    lp.ScalarArg(name, get_dtype(val))
                    for name, val in var_values.iteritems()
                    ])
        ck = lp.CompiledKernel(ctx, knl)
        evt, (lp_value,) = ck(queue, out_host=True, **var_values)
        err = abs(true_value-lp_value)/abs(true_value)
        if abs(err) > 1e-10:
            print "---------------------------------------------------------------------"
            print "WRONG: rel error=%g" % err
            print "true=%r" % true_value
            print "loopy=%r" % lp_value
            print "---------------------------------------------------------------------"
            print ck.code
            print "---------------------------------------------------------------------"
            print var_values
            print "---------------------------------------------------------------------"
            print repr(expr)
            print "---------------------------------------------------------------------"
            print expr
            print "---------------------------------------------------------------------"
            1/0






def test_empty_reduction(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(ctx.devices[0],
            [
                "{[i]: 0<=i<20}",
                "{[j]: 0<=j<0}"
                ],
            [
                "a[i] = sum(j, j)",
                ],
            [
                lp.GlobalArg("a", dtype, (20,)),
                ])
    cknl = lp.CompiledKernel(ctx, knl)

    evt, (a,) = cknl(queue)

    assert (a.get() == 0).all()





def test_nested_dependent_reduction(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(ctx.devices[0],
            [
                "{[i]: 0<=i<20}",
                "{[j]: 0<=j<i+sumlen}"
                ],
            [
                "<> sumlen = l[i]",
                "a[i] = sum(j, j)",
                ],
            [
                lp.GlobalArg("a", dtype, (20,)),
                lp.GlobalArg("l", np.int32, (20,)),
                ])

    cknl = lp.CompiledKernel(ctx, knl)
    cknl.print_code()

    evt, (a,) = cknl(queue)





def test_dependent_loop_bounds(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            [
                "{[i]: 0<=i<n}",
                "{[jj]: 0<=jj<row_len}",
                ],
            [
                "<> row_len = a_rowstarts[i+1] - a_rowstarts[i]",
                "ax[i] = sum(jj, a_values[a_rowstarts[i]+jj])",
                ],
            [
                lp.GlobalArg("a_rowstarts", np.int32),
                lp.GlobalArg("a_indices", np.int32),
                lp.GlobalArg("a_values", dtype),
                lp.GlobalArg("x", dtype),
                lp.GlobalArg("ax", dtype),
                lp.ScalarArg("n", np.int32),
                ],
            assumptions="n>=1 and row_len>=1")

    cknl = lp.CompiledKernel(ctx, knl)
    print "---------------------------------------------------"
    cknl.print_code()
    print "---------------------------------------------------"




def test_dependent_loop_bounds_2(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            [
                "{[i]: 0<=i<n}",
                "{[jj]: 0<=jj<row_len}",
                ],
            [
                "<> row_start = a_rowstarts[i]",
                "<> row_len = a_rowstarts[i+1] - row_start",
                "ax[i] = sum(jj, a_values[row_start+jj])",
                ],
            [
                lp.GlobalArg("a_rowstarts", np.int32),
                lp.GlobalArg("a_indices", np.int32),
                lp.GlobalArg("a_values", dtype),
                lp.GlobalArg("x", dtype),
                lp.GlobalArg("ax", dtype),
                lp.ScalarArg("n", np.int32),
                ],
            assumptions="n>=1 and row_len>=1")

    knl = lp.split_dimension(knl, "i", 128, outer_tag="g.0",
            inner_tag="l.0")
    cknl = lp.CompiledKernel(ctx, knl)
    print "---------------------------------------------------"
    cknl.print_code()
    print "---------------------------------------------------"





def test_dependent_loop_bounds_3(ctx_factory):
    # The point of this test is that it shows a dependency between
    # domains that is exclusively mediated by the row_len temporary.
    # It also makes sure that row_len gets read before any
    # conditionals use it.

    dtype = np.dtype(np.float32)
    ctx = ctx_factory()

    knl = lp.make_kernel(ctx.devices[0],
            [
                "{[i]: 0<=i<n}",
                "{[jj]: 0<=jj<row_len}",
                ],
            [
                "<> row_len = a_row_lengths[i]",
                "a[i,jj] = 1",
                ],
            [
                lp.GlobalArg("a_row_lengths", np.int32),
                lp.GlobalArg("a", dtype, shape=("n,n"), order="C"),
                lp.ScalarArg("n", np.int32),
                ])

    assert knl.parents_per_domain()[1] == 0

    knl = lp.split_dimension(knl, "i", 128, outer_tag="g.0",
            inner_tag="l.0")

    cknl = lp.CompiledKernel(ctx, knl)
    print "---------------------------------------------------"
    cknl.print_code()
    print "---------------------------------------------------"

    knl_bad = lp.split_dimension(knl, "jj", 128, outer_tag="g.1",
            inner_tag="l.1")

    import pytest
    with pytest.raises(RuntimeError):
        list(lp.generate_loop_schedules(knl_bad))




def test_independent_multi_domains(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(ctx.devices[0],
            [
                "{[i]: 0<=i<n}",
                "{[j]: 0<=j<n}",
                ],
            [
                "a[i,j] = 1",
                ],
            [
                lp.GlobalArg("a", dtype, shape=("n,n"), order="C"),
                lp.ScalarArg("n", np.int32),
                ])


    knl = lp.split_dimension(knl, "i", 16, outer_tag="g.0",
            inner_tag="l.0")
    knl = lp.split_dimension(knl, "j", 16, outer_tag="g.1",
            inner_tag="l.1")
    assert knl.parents_per_domain() == 2*[None]

    n = 50
    cknl = lp.CompiledKernel(ctx, knl)
    evt, (a,) = cknl(queue, n=n, out_host=True)

    assert a.shape == (50, 50)
    assert (a == 1).all()





def test_bare_data_dependency(ctx_factory):
    dtype = np.dtype(np.float32)
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(ctx.devices[0],
            [
                "[znirp] -> {[i]: 0<=i<znirp}",
                ],
            [
                "<> znirp = n",
                "a[i] = 1",
                ],
            [
                lp.GlobalArg("a", dtype, shape=("n"), order="C"),
                lp.ScalarArg("n", np.int32),
                ])

    cknl = lp.CompiledKernel(ctx, knl)
    n = 20000
    evt, (a,) = cknl(queue, n=n, out_host=True)

    assert a.shape == (n,)
    assert (a == 1).all()




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

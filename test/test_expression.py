__copyright__ = "Copyright (C) 2019 Andreas Kloeckner"

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

import sys
import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.clmath  # noqa
import pyopencl.clrandom  # noqa
import pytest

from pymbolic.mapper.evaluator import EvaluationMapper


import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

__all__ = [
        "pytest_generate_tests",
        "cl"  # "cl.create_some_context"
        ]


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


# {{{ code generator fuzzing

class BoundsCheckError(ValueError):
    pass


class BoundsCheckingEvaluationMapper(EvaluationMapper):
    def __init__(self, context, lbound, ubound):
        super().__init__(context)
        self.lbound = lbound
        self.ubound = ubound

    def rec(self, expr):
        result = super().rec(expr)

        if result > self.ubound:
            raise BoundsCheckError()
        if result < self.lbound:
            raise BoundsCheckError()

        return result


def make_random_fp_value(use_complex):
    from random import randrange, uniform
    v = randrange(3)
    if v == 0:
        while True:
            z = randrange(-1000, 1000)
            if z:
                return z

    elif v == 1 or not use_complex:
        return uniform(-10, 10)
    else:
        cval = uniform(-10, 10) + 1j*uniform(-10, 10)
        if randrange(0, 2) == 0:
            return np.complex128(cval)
        else:
            return np.complex128(cval)


def make_random_fp_expression(prefix, var_values, size, use_complex):
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
            make_random_fp_expression(prefix, var_values, size, use_complex)
            for i in range(term_count)))
    elif v < 750:
        return make_random_fp_value(use_complex=use_complex)
    elif v < 1000:
        var_name = "var_%s_%d" % (prefix, len(var_values))
        assert var_name not in var_values
        var_values[var_name] = make_random_fp_value(use_complex=use_complex)
        return p.Variable(var_name)
    elif v < 1250:
        # FIXME: What does this comment mean?
        # Cannot use '-' because that destroys numpy constants.
        return p.Sum((
            make_random_fp_expression(prefix, var_values, size, use_complex),
            - make_random_fp_expression(prefix, var_values, size, use_complex)))
    elif v < 1500:
        # FIXME: What does this comment mean?
        # Cannot use '/' because that destroys numpy constants.
        return p.Quotient(
                make_random_fp_expression(prefix, var_values, size, use_complex),
                make_random_fp_expression(prefix, var_values, size, use_complex))
    else:
        raise AssertionError()


def make_random_int_value(nonneg):
    from random import randrange
    if nonneg:
        return randrange(1, 100)

    else:
        while True:
            z = randrange(-100, 100)
            if z:
                return z


def make_random_int_expression(prefix, var_values, size, nonneg):
    from random import randrange
    import pymbolic.primitives as p
    if size[0] < 10:
        v = randrange(800)
    else:
        v = randrange(1000)

    size[0] += 1

    if v < 10:
        var_name = "var_%s_%d" % (prefix, len(var_values))
        assert var_name not in var_values
        var_values[var_name] = make_random_int_value(nonneg)
        return p.Variable(var_name)
    elif v < 200 and size[0] < 40:
        term_count = randrange(2, 5)
        if randrange(2) < 1:
            cls = p.Sum
        else:
            cls = p.Product
        return cls(tuple(
            make_random_int_expression(
                prefix, var_values, size, nonneg=nonneg)
            for i in range(term_count)))
    elif v < 400 and size[0] < 40:
        return p.FloorDiv(
                make_random_int_expression(
                    prefix, var_values, size, nonneg=nonneg),
                make_nonzero_random_int_expression(
                    prefix, var_values, size, nonneg=nonneg))
    elif v < 600 and size[0] < 40:
        return p.Remainder(
                make_random_int_expression(
                    prefix, var_values, size, nonneg=nonneg),
                make_nonzero_random_int_expression(
                    prefix, var_values, size, nonneg=nonneg))
    elif v < 800 and not nonneg and size[0] < 40:
        return p.Sum((
            make_random_int_expression(
                prefix, var_values, size, nonneg=nonneg),
            - make_random_int_expression(
                prefix, var_values, size, nonneg=nonneg),
            ))
    else:
        return make_random_int_value(nonneg)


def make_nonzero_random_int_expression(prefix, var_values, size, nonneg):
    while True:
        var_values_new = var_values.copy()
        size_new = size[:]
        result = make_random_int_expression(
                prefix, var_values_new, size_new, nonneg)

        result_eval = EvaluationMapper(var_values_new)(result)

        if result_eval != 0:
            var_values.update(var_values_new)
            size[:] = size_new

            return result

    raise AssertionError()


def generate_random_fuzz_examples(expr_type):
    i = 0
    while True:
        size = [0]
        var_values = {}
        if expr_type in ["real", "complex"]:
            expr = make_random_fp_expression("e%d" % i, var_values, size,
                    use_complex=expr_type == "complex")
        elif expr_type == "int":
            expr = make_random_int_expression(
                    "e%d" % i, var_values, size, nonneg=False)
        elif expr_type == "int_nonneg":
            expr = make_random_int_expression(
                    "e%d" % i, var_values, size, nonneg=True)
        else:
            raise ValueError("unknown expr_type: %s" % expr_type)

        yield i, expr, var_values

        i += 1


def assert_parse_roundtrip(expr):
    from pymbolic.mapper.stringifier import StringifyMapper
    strified = StringifyMapper()(expr)
    from pymbolic import parse
    parsed_expr = parse(strified)
    print(expr)
    print(parsed_expr)
    assert expr == parsed_expr


@pytest.mark.parametrize("target_cls", [lp.PyOpenCLTarget, lp.ExecutableCTarget])
@pytest.mark.parametrize("random_seed", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("expr_type", ["int", "int_nonneg", "real", "complex"])
def test_fuzz_expression_code_gen(ctx_factory, expr_type, random_seed, target_cls):
    from pymbolic import evaluate

    def get_numpy_type(x):
        if expr_type in ["real", "complex"]:
            if isinstance(x, (complex, np.complexfloating)):
                return np.complex128
            else:
                return np.float64

        elif expr_type in ["int", "int_nonneg"]:
            return np.int64

        else:
            raise ValueError("unknown expr_type: %s" % expr_type)

    from random import seed

    seed(random_seed)

    data = []
    instructions = []

    ref_values = {}

    if expr_type in ["real", "complex"]:
        result_type = np.complex128
    elif expr_type in ["int", "int_nonneg"]:
        result_type = np.int64
    else:
        raise AssertionError()

    var_names = []

    fuzz_iter = iter(generate_random_fuzz_examples(expr_type))
    count = 0

    while True:
        if count == 10:
            break

        i, expr, var_values = next(fuzz_iter)

        var_name = "expr%d" % i

        # print(expr)
        #assert_parse_roundtrip(expr)

        if expr_type in ["int", "int_nonneg"]:
            result_type_iinfo = np.iinfo(np.int32)
            bceval_mapper = BoundsCheckingEvaluationMapper(
                    var_values,
                    lbound=result_type_iinfo.min,
                    ubound=result_type_iinfo.max)
            # print(expr)
            try:
                ref_values[var_name] = bceval_mapper(expr)
            except BoundsCheckError:
                print(expr)
                print("BOUNDS CHECK FAILED")
                continue
        else:
            try:
                ref_values[var_name] = evaluate(expr, var_values)
            except ZeroDivisionError:
                continue

        count += 1

        data.append(lp.GlobalArg(var_name,
            result_type,
            shape=()))
        data.extend([
            lp.TemporaryVariable(name, get_numpy_type(val))
            for name, val in var_values.items()
            ])
        instructions.extend([
            lp.Assignment(name, get_numpy_type(val)(val))
            for name, val in var_values.items()
            ])
        instructions.append(lp.Assignment(var_name, expr))

        if expr_type == "int_nonneg":
            var_names.extend(var_values)

    if issubclass(target_cls, lp.ExecutableCTarget):
        # https://github.com/inducer/loopy/issues/686
        # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107127

        from shutil import which
        gcc_10 = which("gcc-10")
        if gcc_10 is not None:
            from loopy.target.c.c_execution import CCompiler
            target = target_cls(compiler=CCompiler(cc=gcc_10))
        else:
            from warnings import warn
            warn("Using default C compiler because gcc-10 was not found. "
                 "These tests may take a long time, because of "
                 "https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107127.")
            target = target_cls()

    else:
        target = target_cls()

    knl = lp.make_kernel("{ : }", instructions, data, seq_dependencies=True,
            target=target)

    import islpy as isl
    knl = lp.assume(knl, isl.BasicSet(
            "[%s] -> { : %s}"
            % (
                ", ".join(var_names),
                " and ".join("%s >= 0" % name for name in var_names))))

    knl = lp.set_options(knl, return_dict=True)
    # print(knl)

    if type(target) is lp.PyOpenCLTarget:
        cl_ctx = ctx_factory()
        knl = lp.set_options(knl, write_code=True)
        with cl.CommandQueue(cl_ctx) as queue:
            evt, lp_values = knl(queue, out_host=True)
    elif type(target) is lp.ExecutableCTarget:
        evt, lp_values = knl()
    else:
        raise NotImplementedError("unsupported target")

    for name, ref_value in ref_values.items():
        lp_value = lp_values[name]
        if expr_type in ["real", "complex"]:
            err = abs(ref_value-lp_value)/abs(ref_value)
        elif expr_type in ["int", "int_nonneg"]:
            err = abs(ref_value-lp_value)
        else:
            raise AssertionError()

        if abs(err) > 1e-10:
            print(80*"-")
            print(knl)
            print(80*"-")
            print(lp.generate_code_v2(knl).device_code())
            print(80*"-")
            print(f"WRONG: {name} rel error={err:g}")
            print("reference=%r" % ref_value)
            print("loopy=%r" % lp_value)
            print(80*"-")
            1/0

    print(lp.generate_code_v2(knl).device_code())

# }}}


def test_sci_notation_literal(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    set_kernel = lp.make_kernel(
         """ { [i]: 0<=i<12 } """,
         """ out[i] = 1e-12""")

    set_kernel = lp.set_options(set_kernel, write_code=True)

    evt, (out,) = set_kernel(queue)

    assert (np.abs(out.get() - 1e-12) < 1e-20).all()


def test_indexof(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
         """ { [i,j]: 0<=i,j<5 } """,
         """ out[i,j] = indexof(out[i,j])""",
         [lp.GlobalArg("out", is_input=False, shape=lp.auto)]
    )

    knl = lp.set_options(knl, write_code=True)

    (evt, (out,)) = knl(queue)
    out = out.get()

    assert np.array_equal(out.ravel(order="C"), np.arange(25))


def test_indexof_vec(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    if (
            # Accurate as of 2019-11-04
            ctx.devices[0].platform.name.startswith("Intel")):
        pytest.skip("target ICD miscompiles vector code")

    knl = lp.make_kernel(
         """ { [i,j,k]: 0<=i,j,k<4 } """,
         """ out[i,j,k] = indexof_vec(out[i,j,k])""",
         [lp.GlobalArg("out", shape=lp.auto, is_input=False)])

    knl = lp.tag_inames(knl, {"i": "vec"})
    knl = lp.tag_data_axes(knl, "out", "vec,c,c")
    knl = lp.set_options(knl, write_code=True)

    (evt, (out,)) = knl(queue)
    #out = out.get()
    #assert np.array_equal(out.ravel(order="C"), np.arange(25))


def test_is_expression_equal():
    from loopy.symbolic import is_expression_equal
    from pymbolic import var

    x = var("x")
    y = var("y")

    assert is_expression_equal(x+2, 2+x)

    assert is_expression_equal((x+2)**2, x**2 + 4*x + 4)
    assert is_expression_equal((x+y)**2, x**2 + 2*x*y + y**2)


def test_integer_associativity():
    knl = lp.make_kernel(
            "{[i] : 0<=i<arraylen}",
            """
            e := (i // (ncomp * elemsize))
            d := ((i // elemsize) % ncomp)
            s := (i % elemsize)
            v[i] = u[ncomp * indices[(s) + elemsize*(e)] + (d)]
            """)

    knl = lp.add_and_infer_dtypes(
            knl, {"u": np.float64, "elemsize, ncomp, indices": np.int32})
    import islpy as isl
    knl = lp.assume(
            knl, isl.BasicSet("[elemsize, ncomp] -> "
            "{ : elemsize>= 0 and ncomp >= 0}"))
    print(lp.generate_code_v2(knl).device_code())
    assert (
            "u[ncomp * indices[i % elemsize + elemsize "
            "* (i / (ncomp * elemsize))] "
            "+ loopy_mod_pos_b_int32(i / elemsize, ncomp)]"
            in lp.generate_code_v2(knl).device_code())


def test_floor_div():
    knl = lp.make_kernel(
        "{ [i]: 0<=i<10 }",
        "out[i] = (i-1)*(i-2)//2")
    assert "loopy_floor_div" in lp.generate_code_v2(knl).device_code()

    knl = lp.make_kernel(
        "{ [i]: 0<=i<10 }",
        "out[i] = (i*(i+1))//2")
    assert "loopy_floor_div" not in lp.generate_code_v2(knl).device_code()


def test_divide_precedence(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{:}",
            """
            x[0] = c*(a/b)
            y[0] = c*(a%b)
            """,
            [lp.ValueArg("a, b, c", np.int32),
                lp.GlobalArg("x, y", np.int32, shape=lp.auto)])
    print(lp.generate_code_v2(knl).device_code())

    evt, (x_out, y_out) = knl(queue, c=2, b=2, a=5)
    evt.wait()
    assert x_out.get() == 4
    assert y_out.get() == 2


@pytest.mark.parametrize("target", [lp.PyOpenCLTarget, lp.ExecutableCTarget])
def test_complex_support(ctx_factory, target):
    knl = lp.make_kernel(
            "{[i, i1, i2]: 0<=i,i1,i2<10}",
            """
            euler1[i]  = exp(3.14159265359j)
            euler2[i] = (2.7182818284 ** (3.14159265359j))
            euler1_real[i] = real(euler1[i])
            euler1_imag[i] = imag(euler1[i])
            real_times_complex[i] = in1[i]*(in2[i]*1j)
            real_plus_complex[i] = in1[i] + (in2[i]*1j)
            abs_complex[i] = abs(real_plus_complex[i])
            complex_div_complex[i] = (2jf + 7*in1[i])/(32jf + 37*in1[i])
            complex_div_real[i] = (2jf + 7*in1[i])/in1[i]
            real_div_complex[i] = in1[i]/(2jf + 7*in1[i])
            out_sum = sum(i1, 1.0*i1 + i1*1jf)*sum(i2, 1.0*i2 + i2*1jf)
            conj_out_sum = conj(out_sum)
            """,
            [
                lp.GlobalArg("out_sum, euler1, real_plus_complex",
                            is_input=False, shape=lp.auto),
                ...
            ],
            target=target(),
            seq_dependencies=True)
    knl = lp.set_options(knl, return_dict=True)

    n = 10

    in1 = np.random.rand(n)
    in2 = np.random.rand(n)

    kwargs = {"in1": in1, "in2": in2}

    if target == lp.PyOpenCLTarget:
        knl = lp.set_options(knl, write_code=True)
        cl_ctx = ctx_factory()
        with cl.CommandQueue(cl_ctx) as queue:
            evt, out = knl(queue, **kwargs)
    elif target == lp.ExecutableCTarget:
        evt, out = knl(**kwargs)
    else:
        raise NotImplementedError("unsupported target")

    np.testing.assert_allclose(out["euler1"], -1)
    np.testing.assert_allclose(out["euler2"], -1)
    np.testing.assert_allclose(out["euler1_real"], -1)
    np.testing.assert_allclose(out["euler1_imag"], 0, atol=1e-10)
    np.testing.assert_allclose(out["real_times_complex"], in1*(in2*1j))
    np.testing.assert_allclose(out["real_plus_complex"], in1+(in2*1j))
    np.testing.assert_allclose(out["abs_complex"], np.abs(out["real_plus_complex"]))
    np.testing.assert_allclose(out["complex_div_complex"], (2j+7*in1)/(32j+37*in1))
    np.testing.assert_allclose(out["complex_div_real"], (2j + 7*in1)/in1)
    np.testing.assert_allclose(out["real_div_complex"], in1/(2j + 7*in1))
    np.testing.assert_allclose(out["out_sum"], (0.5*n*(n-1) + 0.5*n*(n-1)*1j) ** 2)
    np.testing.assert_allclose(out["conj_out_sum"],
                               (0.5*n*(n-1) - 0.5*n*(n-1)*1j) ** 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_real_with_real_argument(ctx_factory, dtype):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
            "{[i]: 0 <= i < nresult}",
            "result[i] = real(ary[i])",
            )

    rng = np.random.default_rng()
    ary = cl.array.to_device(queue, rng.random(128).astype(dtype))

    _, (result,) = knl(queue, ary=ary)

    assert result.dtype == ary.dtype
    np.testing.assert_allclose(result.get(), np.real(ary.get()))


def test_bool_type_context(ctx_factory):
    # Checks if a boolean type context is correctly handled in codegen phase.
    # See https://github.com/inducer/loopy/pull/258
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{:}",
        """
        k = 8.0 and 7.0
        """,
        [
            lp.GlobalArg("k", dtype=np.bool_, shape=lp.auto),
        ])

    evt, (out,) = knl(queue)
    assert out.get() == np.logical_and(7.0, 8.0)


def test_np_bool_handling(ctx_factory):
    import pymbolic.primitives as p
    from loopy.symbolic import parse
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = lp.make_kernel(
        "{:}",
        [lp.Assignment(parse("y"), p.LogicalNot(np.bool_(False)))],
        [lp.GlobalArg("y", dtype=np.bool_, shape=lp.auto)])
    evt, (out,) = knl(queue)
    assert out.get().item() is True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

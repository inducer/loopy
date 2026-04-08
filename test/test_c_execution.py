__copyright__ = "Copyright (C) 2017 Nick Curtis"

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

import logging

import numpy as np
import pytest

import loopy as lp
from loopy import CACHING_ENABLED
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


logger = logging.getLogger(__name__)


def test_c_target():
    from loopy.target.c import ExecutableCTarget

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = 2*a[i]",
            [
                lp.GlobalArg("out", np.float32, shape=lp.auto),
                lp.GlobalArg("a", np.float32, shape=lp.auto),
                "..."
                ],
            target=ExecutableCTarget())

    assert np.allclose(knl(a=np.arange(16, dtype=np.float32))[1],
                2 * np.arange(16, dtype=np.float32))


def test_c_target_strides():
    from loopy.target.c import ExecutableCTarget

    def __get_kernel(order="C"):
        return lp.make_kernel(
                "{ [i,j]: 0<=i,j<n }",
                "out[i, j] = 2*a[i, j]",
                [
                    lp.GlobalArg("out", np.float32, shape=("n", "n"), order=order),
                    lp.GlobalArg("a", np.float32, shape=("n", "n"), order=order),
                    "..."
                    ],
                target=ExecutableCTarget())

    # test with C-order
    knl = __get_kernel("C")
    a_np = np.reshape(np.arange(16 * 16, dtype=np.float32), (16, -1),
                      order="C")

    assert np.allclose(knl(a=a_np)[1],
                2 * a_np)

    # test with F-order
    knl = __get_kernel("F")
    a_np = np.reshape(np.arange(16 * 16, dtype=np.float32), (16, -1),
                      order="F")

    assert np.allclose(knl(a=a_np)[1],
                2 * a_np)


def test_c_target_strides_nonsquare():
    from loopy.target.c import ExecutableCTarget
    rng = np.random.default_rng(seed=42)

    def __get_kernel(order="C"):
        indices = ["i", "j", "k"]
        sizes = tuple(rng.integers(1, 11, size=len(indices)))
        # create domain strings
        domain_template = "{{ [{iname}]: 0 <= {iname} < {size} }}"
        domains = []
        for idx, size in zip(indices, sizes, strict=True):
            domains.append(domain_template.format(
                iname=idx,
                size=size))
        statement = "out[{indexed}] = 2 * a[{indexed}]".format(
            indexed=", ".join(indices))
        return lp.make_kernel(
                domains,
                statement,
                [
                    lp.GlobalArg("out", np.float32, shape=sizes, order=order),
                    lp.GlobalArg("a", np.float32, shape=sizes, order=order),
                    "..."
                    ],
                target=ExecutableCTarget(),
                name="nonsquare_strides")

    # test with C-order
    knl = __get_kernel("C")
    a_lp = next(x for x in knl["nonsquare_strides"].args if x.name == "a")
    a_np = np.reshape(np.arange(np.prod(a_lp.shape), dtype=np.float32),
                      a_lp.shape,
                      order="C")

    assert np.allclose(knl(a=a_np)[1],
                2 * a_np)

    # test with F-order
    knl = __get_kernel("F")
    a_lp = next(x for x in knl["nonsquare_strides"].args if x.name == "a")
    a_np = np.reshape(np.arange(np.prod(a_lp.shape), dtype=np.float32),
                      a_lp.shape,
                      order="F")

    assert np.allclose(knl(a=a_np)[1],
                2 * a_np)


def test_c_optimizations():
    from loopy.target.c import ExecutableCTarget

    rng = np.random.default_rng(seed=42)

    def __get_kernel(order="C"):
        indices = ["i", "j", "k"]
        sizes = tuple(rng.integers(1, 11, size=len(indices)))
        # create domain strings
        domain_template = "{{ [{iname}]: 0 <= {iname} < {size} }}"
        domains = []
        for idx, size in zip(indices, sizes, strict=True):
            domains.append(domain_template.format(
                iname=idx,
                size=size))
        statement = "out[{indexed}] = 2 * a[{indexed}]".format(
            indexed=", ".join(indices))
        return lp.make_kernel(
                domains,
                statement,
                [
                    lp.GlobalArg("out", np.float32, shape=sizes, order=order),
                    lp.GlobalArg("a", np.float32, shape=sizes, order=order),
                    "..."
                    ],
                target=ExecutableCTarget()), sizes

    # test with ILP
    knl, sizes = __get_kernel("C")
    knl = lp.split_iname(knl, "i", 4, inner_tag="ilp")
    a_np = np.reshape(np.arange(np.prod(sizes), dtype=np.float32),
                      sizes,
                      order="C")

    assert np.allclose(knl(a=a_np)[1], 2 * a_np)

    # test with unrolling
    knl, sizes = __get_kernel("C")
    knl = lp.split_iname(knl, "i", 4, inner_tag="unr")
    a_np = np.reshape(np.arange(np.prod(sizes), dtype=np.float32),
                      sizes,
                      order="C")

    assert np.allclose(knl(a=a_np)[1], 2 * a_np)


def test_function_decl_extractor():
    # ensure that we can tell the difference between pointers, constants, etc.
    # in execution
    from loopy.target.c import ExecutableCTarget

    knl = lp.make_kernel("{[i]: 0 <= i < 10}",
        """
            a[i] = b[i] + v
        """,
        [lp.GlobalArg("a", shape=(10,), dtype=np.int32),
         lp.ConstantArg("b", shape=(10)),
         lp.ValueArg("v", dtype=np.int32)],
        target=ExecutableCTarget())

    assert np.allclose(knl(b=np.arange(10), v=-1)[1], np.arange(10) - 1)


@pytest.mark.skipif(not CACHING_ENABLED, reason="Can't test caching when disabled")
def test_c_caching():
    # ensure that codepy is correctly caching the code
    from loopy.target.c import ExecutableCTarget

    class TestingLogger:
        def start_capture(self, loglevel=logging.DEBUG):
            """ Start capturing log output to a string buffer.
                @param newLogLevel: Optionally change the global logging level, e.g.
                logging.DEBUG
            """
            from io import StringIO
            self.buffer = StringIO()
            self.buffer.write("Log output")

            logger = logging.getLogger()
            if loglevel:
                self.oldloglevel = logger.getEffectiveLevel()
                logger.setLevel(loglevel)
            else:
                self.oldloglevel = None

            self.loghandler = logging.StreamHandler(self.buffer)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s "
                                          "- %(message)s")
            self.loghandler.setFormatter(formatter)
            logger.addHandler(self.loghandler)

        def stop_capture(self):
            """ Stop capturing log output.

            @return: Collected log output as string
            """

            # Remove our handler
            logger = logging.getLogger()

            # Restore logging level (if any)
            if self.oldloglevel is not None:
                logger.setLevel(self.oldloglevel)
            logger.removeHandler(self.loghandler)

            self.loghandler.flush()
            self.buffer.flush()

            return self.buffer.getvalue()

    def __get_knl():
        return lp.make_kernel("{[i]: 0 <= i < 10}",
        """
            a[i] = b[i]
        """,
        [lp.GlobalArg("a", shape=(10,), dtype=np.int32),
         lp.ConstantArg("b", shape=(10))],
                             target=ExecutableCTarget(),
                             name="cache_test")

    knl = __get_knl()
    # compile
    assert np.allclose(knl(b=np.arange(10))[1], np.arange(10))
    # setup test logger to check logs
    tl = TestingLogger()
    tl.start_capture()
    # copy kernel such that we share the same executor cache
    knl = knl.copy()
    # but use different args, so we can't cache the result
    assert np.allclose(knl(b=np.arange(1, 11))[1], np.arange(1, 11))
    # and get logs
    logs = tl.stop_capture()
    # check that we didn't recompile
    assert "Kernel cache_test retrieved from cache" in logs


def test_c_execution_with_global_temporaries():
    # ensure that the "host" code of a bare ExecutableCTarget with
    # global constant temporaries is None

    from loopy.target.c import ExecutableCTarget
    AS = lp.AddressSpace        # noqa: N806
    n = 10

    knl = lp.make_kernel("{[i]: 0 <= i < n}",
        """
            a[i] = b[i]
        """,
        [lp.GlobalArg("a", shape=(n,), dtype=np.int32),
         lp.TemporaryVariable("b", shape=(n,),
                              initializer=np.arange(n, dtype=np.int32),
                              dtype=np.int32,
                              read_only=True,
                              address_space=AS.GLOBAL)],
        target=ExecutableCTarget())

    knl = lp.fix_parameters(knl, n=n)
    assert ("int b[%d]" % n) not in lp.generate_code_v2(knl).host_code()
    assert np.allclose(knl(a=np.zeros(10, dtype=np.int32))[1], np.arange(10))


def test_missing_compilers():
    from codepy.toolchain import GCCToolchain

    from loopy.target.c import CTarget, ExecutableCTarget
    from loopy.target.c.c_execution import CCompiler

    def __test(evalfunc, target, **targetargs):
        n = 10

        knl = lp.make_kernel("{[i]: 0 <= i < n}",
            """
                a[i] = b[i]
            """,
            [lp.GlobalArg("a", shape=(n,), dtype=np.int32),
             lp.GlobalArg("b", shape=(n,), dtype=np.int32)],
            target=target(**targetargs))

        knl = lp.fix_parameters(knl, n=n)
        return evalfunc(knl)

    assert __test(lambda knl: lp.generate_code_v2(knl).device_code(), CTarget)

    from pytools.prefork import ExecError

    def eval_tester(knl):
        return np.allclose(knl(a=np.zeros(10, dtype=np.int32),
                               b=np.arange(10, dtype=np.int32))[1], np.arange(10))
    import os
    path_store = os.environ["PATH"]
    ccomp = None
    try:
        # test with path wiped out such that we can't find gcc
        with pytest.raises(ExecError):
            os.environ["PATH"] = ""
            ccomp = CCompiler()
            __test(eval_tester, ExecutableCTarget, compiler=ccomp)
    finally:
        # make sure we restore the path
        os.environ["PATH"] = path_store
        # and, with the path restored we should now be able to properly execute with
        # the default (non-guessed) toolchain!
        __test(eval_tester, ExecutableCTarget, compiler=ccomp)

    # next test that some made up compiler can be specified
    ccomp = CCompiler(cc="foo")
    assert isinstance(ccomp.toolchain, GCCToolchain)
    assert ccomp.toolchain.cc == "foo"

    # and that said made up compiler errors out

    with pytest.raises(ExecError):
        __test(eval_tester, ExecutableCTarget, compiler=ccomp)


def test_one_length_loop():
    # https://github.com/inducer/loopy/issues/239
    knl = lp.make_kernel(
            "{[i]: 0<=i<1}",
            """
            a[i] = 42.0
            """, target=lp.ExecutableCTarget())

    _, (out, ) = knl()
    assert out == 42


def test_scalar_global_args():
    n = np.random.default_rng().integers(30, 100)
    _evt, (out,) = lp.make_kernel(
            "{[i]: 0<=i<n}",
            "res  = sum(i, i)",
            target=lp.ExecutableCTarget())(n=n)

    assert out == (n*(n-1)/2)  # noqa: RUF069


@pytest.mark.parametrize("signed_dtype,unsigned_dtype", [
    (np.int8, np.uint8),
    (np.int16, np.uint16),
    (np.int32, np.uint32),
    (np.int64, np.uint64),
])
def test_mixed_sign_comparison(signed_dtype, unsigned_dtype):
    """Mixed-sign comparisons must follow Python semantics (not C's
    implicit unsigned-promotion rule).  The classic C footgun is:

        int x = -1; unsigned int y = 1;
        if (x < y) { ... }  // NOT taken — x becomes UINT_MAX

    Loopy should generate explicit casts so the branch IS taken.
    """
    t_unit = lp.make_kernel(
        "{:}",
        """
        lt_result   = if(sv < uv, 1, 0)
        le_result   = if(sv <= uv, 1, 0)
        gt_result   = if(sv > uv, 1, 0)
        ge_result   = if(sv >= uv, 1, 0)
        eq_result   = if(sv == uv, 1, 0)
        ne_result   = if(sv != uv, 1, 0)
        """,
        [
            lp.ValueArg("sv", signed_dtype),
            lp.ValueArg("uv", unsigned_dtype),
            lp.GlobalArg("lt_result, le_result, gt_result, "
                         "ge_result, eq_result, ne_result",
                         np.int32, shape=lp.auto),
        ],
        target=lp.ExecutableCTarget(),
    )
    t_unit = lp.set_options(t_unit, return_dict=True)

    # --- sv = -1, uv = 1 -------------------------------------------------------
    # Python:  -1 < 1  → True, -1 <= 1 → True, -1 > 1 → False,
    #          -1 >= 1 → False, -1 == 1 → False, -1 != 1 → True
    sv = signed_dtype(-1)
    uv = unsigned_dtype(1)
    _evt, out = t_unit(sv=sv, uv=uv)
    assert out["lt_result"][()] == 1, f"{sv} < {uv} should be True"
    assert out["le_result"][()] == 1, f"{sv} <= {uv} should be True"
    assert out["gt_result"][()] == 0, f"{sv} > {uv} should be False"
    assert out["ge_result"][()] == 0, f"{sv} >= {uv} should be False"
    assert out["eq_result"][()] == 0, f"{sv} == {uv} should be False"
    assert out["ne_result"][()] == 1, f"{sv} != {uv} should be True"

    # --- sv = 1, uv = 1 --------------------------------------------------------
    sv = signed_dtype(1)
    uv = unsigned_dtype(1)
    _evt, out = t_unit(sv=sv, uv=uv)
    assert out["lt_result"][()] == 0, f"{sv} < {uv} should be False"
    assert out["le_result"][()] == 1, f"{sv} <= {uv} should be True"
    assert out["gt_result"][()] == 0, f"{sv} > {uv} should be False"
    assert out["ge_result"][()] == 1, f"{sv} >= {uv} should be True"
    assert out["eq_result"][()] == 1, f"{sv} == {uv} should be True"
    assert out["ne_result"][()] == 0, f"{sv} != {uv} should be False"

    # --- sv = 2, uv = 1 --------------------------------------------------------
    sv = signed_dtype(2)
    uv = unsigned_dtype(1)
    _evt, out = t_unit(sv=sv, uv=uv)
    assert out["lt_result"][()] == 0, f"{sv} < {uv} should be False"
    assert out["le_result"][()] == 0, f"{sv} <= {uv} should be False"
    assert out["gt_result"][()] == 1, f"{sv} > {uv} should be True"
    assert out["ge_result"][()] == 1, f"{sv} >= {uv} should be True"
    assert out["eq_result"][()] == 0, f"{sv} == {uv} should be False"
    assert out["ne_result"][()] == 1, f"{sv} != {uv} should be True"


@pytest.mark.parametrize("signed_dtype,unsigned_dtype", [
    (np.int8, np.uint8),
    (np.int16, np.uint16),
    (np.int32, np.uint32),
    (np.int64, np.uint64),
])
def test_mixed_sign_subtraction(signed_dtype, unsigned_dtype):
    """Mixed-sign subtraction must follow Python semantics.

    In C, ``(int32_t)-1 - (uint32_t)1`` is computed as
    ``(uint32_t)UINT_MAX - 1 = 4294967294``, which when cast to int64
    gives 4294967294, not -2.  Loopy must insert explicit casts so the
    result is -2.
    """
    # result dtype: the narrowest signed type wide enough to hold both
    result_dtype = np.result_type(signed_dtype, unsigned_dtype)
    if result_dtype.kind != "i":
        # int64 vs uint64: numpy promotes to float64; use int64 instead
        result_dtype = np.dtype(np.int64)

    t_unit = lp.make_kernel(
        "{:}",
        """
        diff_sv_uv = sv - uv
        diff_uv_sv = uv - sv
        """,
        [
            lp.ValueArg("sv", signed_dtype),
            lp.ValueArg("uv", unsigned_dtype),
            lp.GlobalArg("diff_sv_uv", result_dtype, shape=lp.auto),
            lp.GlobalArg("diff_uv_sv", result_dtype, shape=lp.auto),
        ],
        target=lp.ExecutableCTarget(),
    )
    t_unit = lp.set_options(t_unit, return_dict=True)

    # --- sv = -1, uv = 1 → expected -2 / 2 ------------------------------------
    sv = signed_dtype(-1)
    uv = unsigned_dtype(1)
    _evt, out = t_unit(sv=sv, uv=uv)
    assert out["diff_sv_uv"][()] == -2, \
        f"{sv} - {uv} should be -2, got {out['diff_sv_uv'][()]}"
    assert out["diff_uv_sv"][()] == 2, \
        f"{uv} - {sv} should be 2, got {out['diff_uv_sv'][()]}"

    # --- sv = 5, uv = 3 → expected 2 / -2 -------------------------------------
    sv = signed_dtype(5)
    uv = unsigned_dtype(3)
    _evt, out = t_unit(sv=sv, uv=uv)
    assert out["diff_sv_uv"][()] == 2, \
        f"{sv} - {uv} should be 2, got {out['diff_sv_uv'][()]}"
    assert out["diff_uv_sv"][()] == -2, \
        f"{uv} - {sv} should be -2, got {out['diff_uv_sv'][()]}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

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

import numpy as np
import loopy as lp
import sys
import pytest
from loopy import CACHING_ENABLED

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa


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

    def __get_kernel(order="C"):
        indicies = ["i", "j", "k"]
        sizes = tuple(np.random.randint(1, 11, size=len(indicies)))
        # create domain strings
        domain_template = "{{ [{iname}]: 0 <= {iname} < {size} }}"
        domains = []
        for idx, size in zip(indicies, sizes):
            domains.append(domain_template.format(
                iname=idx,
                size=size))
        statement = "out[{indexed}] = 2 * a[{indexed}]".format(
            indexed=", ".join(indicies))
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

    def __get_kernel(order="C"):
        indicies = ["i", "j", "k"]
        sizes = tuple(np.random.randint(1, 11, size=len(indicies)))
        # create domain strings
        domain_template = "{{ [{iname}]: 0 <= {iname} < {size} }}"
        domains = []
        for idx, size in zip(indicies, sizes):
            domains.append(domain_template.format(
                iname=idx,
                size=size))
        statement = "out[{indexed}] = 2 * a[{indexed}]".format(
            indexed=", ".join(indicies))
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
    AS = lp.AddressSpace        # noqa
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
    from loopy.target.c import ExecutableCTarget, CTarget
    from loopy.target.c.c_execution import CCompiler
    from codepy.toolchain import GCCToolchain

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

    # and test that we will fail if we remove a required attribute
    del ccomp.toolchain.undefines
    with pytest.raises(AttributeError):
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
    evt, (out,) = lp.make_kernel(
            "{[i]: 0<=i<n}",
            "res  = sum(i, i)",
            target=lp.ExecutableCTarget())(n=n)

    assert out == (n*(n-1)/2)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

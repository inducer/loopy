from __future__ import division, absolute_import, print_function

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

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


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

    def __get_kernel(order='C'):
        return lp.make_kernel(
                "{ [i,j]: 0<=i,j<n }",
                "out[i, j] = 2*a[i, j]",
                [
                    lp.GlobalArg("out", np.float32, shape=('n', 'n'), order=order),
                    lp.GlobalArg("a", np.float32, shape=('n', 'n'), order=order),
                    "..."
                    ],
                target=ExecutableCTarget())

    # test with C-order
    knl = __get_kernel('C')
    a_np = np.reshape(np.arange(16 * 16, dtype=np.float32), (16, -1),
                      order='C')

    assert np.allclose(knl(a=a_np)[1],
                2 * a_np)

    # test with F-order
    knl = __get_kernel('F')
    a_np = np.reshape(np.arange(16 * 16, dtype=np.float32), (16, -1),
                      order='F')

    assert np.allclose(knl(a=a_np)[1],
                2 * a_np)


def test_c_target_strides_nonsquare():
    from loopy.target.c import ExecutableCTarget

    def __get_kernel(order='C'):
        indicies = ['i', 'j', 'k']
        sizes = tuple(np.random.randint(1, 11, size=len(indicies)))
        # create domain strings
        domain_template = '{{ [{iname}]: 0 <= {iname} < {size} }}'
        domains = []
        for idx, size in zip(indicies, sizes):
            domains.append(domain_template.format(
                iname=idx,
                size=size))
        statement = 'out[{indexed}] = 2 * a[{indexed}]'.format(
            indexed=', '.join(indicies))
        return lp.make_kernel(
                domains,
                statement,
                [
                    lp.GlobalArg("out", np.float32, shape=sizes, order=order),
                    lp.GlobalArg("a", np.float32, shape=sizes, order=order),
                    "..."
                    ],
                target=ExecutableCTarget())

    # test with C-order
    knl = __get_kernel('C')
    a_lp = next(x for x in knl.args if x.name == 'a')
    a_np = np.reshape(np.arange(np.product(a_lp.shape), dtype=np.float32),
                      a_lp.shape,
                      order='C')

    assert np.allclose(knl(a=a_np)[1],
                2 * a_np)

    # test with F-order
    knl = __get_kernel('F')
    a_lp = next(x for x in knl.args if x.name == 'a')
    a_np = np.reshape(np.arange(np.product(a_lp.shape), dtype=np.float32),
                      a_lp.shape,
                      order='F')

    assert np.allclose(knl(a=a_np)[1],
                2 * a_np)


def test_c_optimizations():
    from loopy.target.c import ExecutableCTarget

    def __get_kernel(order='C'):
        indicies = ['i', 'j', 'k']
        sizes = tuple(np.random.randint(1, 11, size=len(indicies)))
        # create domain strings
        domain_template = '{{ [{iname}]: 0 <= {iname} < {size} }}'
        domains = []
        for idx, size in zip(indicies, sizes):
            domains.append(domain_template.format(
                iname=idx,
                size=size))
        statement = 'out[{indexed}] = 2 * a[{indexed}]'.format(
            indexed=', '.join(indicies))
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
    knl, sizes = __get_kernel('C')
    knl = lp.split_iname(knl, 'i', 4, inner_tag='ilp')
    a_np = np.reshape(np.arange(np.product(sizes), dtype=np.float32),
                      sizes,
                      order='C')

    assert np.allclose(knl(a=a_np)[1], 2 * a_np)

    # test with unrolling
    knl, sizes = __get_kernel('C')
    knl = lp.split_iname(knl, 'i', 4, inner_tag='unr')
    a_np = np.reshape(np.arange(np.product(sizes), dtype=np.float32),
                      sizes,
                      order='C')

    assert np.allclose(knl(a=a_np)[1], 2 * a_np)


def test_function_decl_extractor():
    # ensure that we can tell the difference between pointers, constants, etc.
    # in execution
    from loopy.target.c import ExecutableCTarget

    knl = lp.make_kernel('{[i]: 0 <= i < 10}',
        """
            a[i] = b[i] + v
        """,
        [lp.GlobalArg('a', shape=(10,), dtype=np.int32),
         lp.ConstantArg('b', shape=(10)),
         lp.ValueArg('v', dtype=np.int32)],
        target=ExecutableCTarget())

    assert np.allclose(knl(b=np.arange(10), v=-1)[1], np.arange(10) - 1)


def test_c_execution_with_global_temporaries():
    # ensure that the "host" code of a bare ExecutableCTarget with
    # global constant temporaries is None

    from loopy.target.c import ExecutableCTarget
    from loopy.kernel.data import temp_var_scope as scopes
    n = 10

    knl = lp.make_kernel('{[i]: 0 <= i < n}',
        """
            a[i] = b[i]
        """,
        [lp.GlobalArg('a', shape=(n,), dtype=np.int32),
         lp.TemporaryVariable('b', shape=(n,),
                              initializer=np.arange(n, dtype=np.int32),
                              dtype=np.int32,
                              read_only=True,
                              scope=scopes.GLOBAL)],
        target=ExecutableCTarget())

    knl = lp.fix_parameters(knl, n=n)
    assert ('int b[%d]' % n) not in lp.generate_code_v2(knl).host_code()
    assert np.allclose(knl(a=np.zeros(10, dtype=np.int32))[1], np.arange(10))

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
import pytest

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


def test_c_target():
    from loopy.target.ispc import ISPCTarget

    knl = lp.make_kernel(
            "{ [i]: 0<=i<n }",
            "out[i] = 2*a[i]",
            [
                lp.GlobalArg("out", np.float32, shape=lp.auto),
                lp.GlobalArg("a", np.float32, shape=lp.auto),
                "..."
                ],
            target=ISPCTarget())

    assert np.allclose(knl(a=np.arange(16, dtype=np.float32))[1],
                2 * np.arange(16, dtype=np.float32))


def test_c_target_strides():
    from loopy.target.ispc import ISPCTarget

    def __get_kernel(order='C'):
        return lp.make_kernel(
                "{ [i,j]: 0<=i,j<n }",
                "out[i, j] = 2*a[i, j]",
                [
                    lp.GlobalArg("out", np.float32, shape=('n', 'n'), order=order),
                    lp.GlobalArg("a", np.float32, shape=('n', 'n'), order=order),
                    "..."
                    ],
                target=ISPCTarget())

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
    from loopy.target.ispc import ISPCTarget

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
                target=ISPCTarget())

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
    from loopy.target.ispc import ISPCTarget

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
                target=ISPCTarget()), sizes

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

    # test with vectorization
    knl, sizes = __get_kernel('C')
    knl = lp.split_iname(knl, 'i', 4, inner_tag='l.0', outer_tag='g.0')
    a_np = np.reshape(np.arange(np.product(sizes), dtype=np.float32),
                      sizes,
                      order='C')

    assert np.allclose(knl(a=a_np)[1], 2 * a_np)


@pytest.mark.parametrize('vec_width', [4, 8, 16])
@pytest.mark.parametrize('target', ['sse2', 'sse4', 'avx1', 'av2'])
@pytest.mark.parametrize('n', [10, 100])
def test_ispc_vector_sizes_and_targets(vec_width, target, n):
    from loopy.target.ispc import ISPCTarget
    from loopy.target.ispc_execution import ISPCCompiler

    compiler = ISPCCompiler(vector_width=vec_width, target=target)
    target = ISPCTarget(compiler=compiler)

    knl = lp.make_kernel(
            '{[i]: 0<=i<n}',
            """
            out[i] = 2 * a[i]
            """,
            [lp.GlobalArg("a", shape=(n,)),
             lp.GlobalArg("out", shape=(n,))],
            target=target)

    knl = lp.fix_parameters(knl, n=n)

    a_np = np.arange(n)
    from loopy import LoopyError
    try:
        assert np.allclose(knl(a=a_np)[1], 2 * a_np)
    except LoopyError as e:
        assert str(e) == "Unexpected expression type: NoneType" and n == 10

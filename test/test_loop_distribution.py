__copyright__ = "Copyright (C) 2021 Kaushik Kulkarni"

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
import pyopencl as cl
import loopy as lp
from loopy.transform.loop_distribution import IllegalLoopDistributionError
import pytest

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

from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa

__all__ = [
        "pytest_generate_tests",
        "cl"  # "cl.create_some_context"
        ]


def test_hello_loop_distribution(ctx_factory):
    ctx = ctx_factory()

    t_unit = lp.make_kernel(
        "{[i,j]: 0<=i, j<10}",
        """
        for i
            a[i] = 10             {id=w_a}
            for j
                b[i, j] = j*a[i]  {id=w_b}
            end
            c[i] = 2*b[i, 5]      {id=w_c}
        end
        """,
        seq_dependencies=True)

    ref_t_unit = t_unit

    knl = lp.distribute_loops(t_unit.default_entrypoint,
                              insn_match="id:w_b",
                              outer_inames=frozenset())
    t_unit = t_unit.with_kernel(knl)
    assert not (knl.id_to_insn["w_a"].within_inames
                & knl.id_to_insn["w_b"].within_inames)
    assert not (knl.id_to_insn["w_c"].within_inames
                & knl.id_to_insn["w_b"].within_inames)
    assert not (knl.id_to_insn["w_a"].within_inames
                & knl.id_to_insn["w_c"].within_inames)

    lp.auto_test_vs_ref(ref_t_unit, ctx, t_unit)


def test_soundness_check(ctx_factory):
    ctx = ctx_factory()

    # {{{ WAW deps

    tunit = lp.make_kernel(
        "{[i]: 1<=i<10}",
        """
        a[i] = i       {id=first_w_a}
        a[i-1] = i**2  {id=second_w_a, dep=first_w_a}
        """
    )
    ref_tunit = tunit

    knl = lp.distribute_loops(tunit.default_entrypoint,
                              "id:second_w_a",
                              outer_inames=frozenset())
    tunit = tunit.with_kernel(knl)
    assert not (knl.id_to_insn["first_w_a"].within_inames
                & knl.id_to_insn["second_w_a"].within_inames)
    lp.auto_test_vs_ref(ref_tunit, ctx, tunit)

    tunit = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
        a[i] = i       {id=first_w_a}
        a[i+1] = i**2  {id=second_w_a, dep=first_w_a}
        """
    )

    with pytest.raises(IllegalLoopDistributionError):
        lp.distribute_loops(tunit.default_entrypoint,
                            "id:second_w_a",
                            outer_inames=frozenset())

    # }}}

    # {{{ RAW deps

    tunit = lp.make_kernel(
        "{[i]: 1<=i<10}",
        """
        b[0] = 0        {id=first_w_b}
        a[i] = i        {id=first_w_a}
        b[i] = 2*a[i-1] {id=second_w_b}
        """,
        seq_dependencies=True,
    )
    ref_tunit = tunit

    knl = lp.distribute_loops(tunit.default_entrypoint,
                              "id:second_w_b",
                              outer_inames=frozenset())
    tunit = tunit.with_kernel(knl)
    assert not (knl.id_to_insn["first_w_a"].within_inames
                & knl.id_to_insn["second_w_b"].within_inames)
    lp.auto_test_vs_ref(ref_tunit, ctx, tunit)

    tunit = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
        a[i] = i        {id=first_w_a}
        b[i] = 2*a[i+1] {id=first_w_b}
        """,
        seq_dependencies=True
    )

    with pytest.raises(IllegalLoopDistributionError):
        lp.distribute_loops(tunit.default_entrypoint,
                            "id:first_w_b",
                            outer_inames=frozenset())

    # }}}

    # {{{ WAR deps

    tunit = lp.make_kernel(
        "{[i, j]: 0<=i<10 and 0<=j<11}",
        """
        b[j] = j**2
        a[i] = b[i+1]   {id=first_w_a}
        b[i] = 2*a[i] {id=first_w_b}
        """,
        seq_dependencies=True
    )
    ref_tunit = tunit

    knl = lp.distribute_loops(tunit.default_entrypoint,
                              "id:first_w_b",
                              outer_inames=frozenset())
    tunit = tunit.with_kernel(knl)
    assert not (knl.id_to_insn["first_w_a"].within_inames
                & knl.id_to_insn["first_w_b"].within_inames)
    lp.auto_test_vs_ref(ref_tunit, ctx, tunit)

    tunit = lp.make_kernel(
        "{[i]: 1<=i<10}",
        """
        b[0] = 0        {id=first_w_b}
        a[i] = b[i-1]   {id=first_w_a}
        b[i] = 2*a[i]   {id=second_w_b}
        """,
        seq_dependencies=True,
    )

    with pytest.raises(IllegalLoopDistributionError):
        lp.distribute_loops(tunit.default_entrypoint,
                            "id:second_w_b",
                            outer_inames=frozenset())

    # }}}


def test_reduction_inames_get_duplicated(ctx_factory):
    ctx = ctx_factory()

    tunit = lp.make_kernel(
        "{[i, j]: 0<=i<100 and 0<=j<10}",
        """
        out1[i] = sum(j, mat1[j] * x1[i, j])  {id=w_out1}
        out2[i] = sum(j, mat2[j] * x2[i, j])  {id=w_out2}
        """,
    )
    tunit = lp.add_dtypes(tunit, {"mat1": np.float64,
                                  "mat2": np.float64,
                                  "x1": np.float64,
                                  "x2": np.float64,
                                  })

    ref_tunit = tunit

    knl = lp.distribute_loops(tunit.default_entrypoint,
                              "id:w_out2",
                              outer_inames=frozenset())
    tunit = tunit.with_kernel(knl)

    assert not (knl.id_to_insn["w_out1"].within_inames
                & knl.id_to_insn["w_out2"].within_inames)
    assert not (knl.id_to_insn["w_out1"].reduction_inames()
                & knl.id_to_insn["w_out2"].reduction_inames())
    lp.auto_test_vs_ref(ref_tunit, ctx, tunit)


def test_avoids_unnecessary_loop_distribution(ctx_factory):
    ctx = ctx_factory()
    tunit = lp.make_kernel(
        "{[i]: 0 <= i < 10}",
        """
        y0[i] = i              {id=w_y0}
        y1[i] = i**2           {id=w_y1}
        y2[i] = y0[i] + i**3   {id=w_y2}
        y3[i] = 2*y2[i]        {id=w_y3}
        y4[i] = i**4 + y1[i]   {id=w_y4}
        """)
    ref_tunit = tunit

    knl = lp.distribute_loops(tunit.default_entrypoint,
                              insn_match="writes:y2 or writes:y4",
                              outer_inames=frozenset())
    tunit = tunit.with_kernel(knl)

    assert (knl.id_to_insn["w_y2"].within_inames
            == knl.id_to_insn["w_y4"].within_inames)
    lp.auto_test_vs_ref(ref_tunit, ctx, tunit)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker

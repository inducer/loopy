from __future__ import division, print_function

__copyright__ = "Copyright (C) 2019 James Stevens"

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

import six  # noqa: F401
import sys
import numpy as np
import loopy as lp
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl
    as pytest_generate_tests)
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa
import logging
from loopy import (
    preprocess_kernel,
    get_one_linearized_kernel,
)
from loopy.schedule.checker.schedule import (
    LEX_VAR_PREFIX,
    STATEMENT_VAR_NAME,
    LTAG_VAR_NAMES,
    GTAG_VAR_NAMES,
)

logger = logging.getLogger(__name__)


# {{{ test pairwise schedule creation

def _lex_space_string(dim_vals, lid_axes=[], gid_axes=[]):
    # Return a string describing lex space dimension assignments
    # (used to create maps below)

    lid_names = [LTAG_VAR_NAMES[i] for i in lid_axes]
    gid_names = [GTAG_VAR_NAMES[i] for i in gid_axes]

    return ", ".join(
        ["%s%d=%s" % (LEX_VAR_PREFIX, idx, str(val))
        for idx, val in enumerate(dim_vals)] + lid_names + gid_names)


def test_pairwise_schedule_creation():
    import islpy as isl
    from loopy.schedule.checker import (
        get_schedules_for_statement_pairs,
    )
    from loopy.schedule.checker.utils import (
        ensure_dim_names_match_and_align,
    )

    # example kernel
    # insn_c depends on insn_b only to create deterministic order
    # insn_d depends on insn_c only to create deterministic order
    knl = lp.make_kernel(
        [
            "{[i]: 0<=i<pi}",
            "{[k]: 0<=k<pk}",
            "{[j]: 0<=j<pj}",
            "{[t]: 0<=t<pt}",
        ],
        """
        for i
            for k
                <>temp = b[i,k]  {id=insn_a}
            end
            for j
                a[i,j] = temp + 1  {id=insn_b,dep=insn_a}
                c[i,j] = d[i,j]  {id=insn_c,dep=insn_b}
            end
        end
        for t
            e[t] = f[t]  {id=insn_d, dep=insn_c}
        end
        """,
        name="example",
        assumptions="pi,pj,pk,pt >= 1",
        )
    knl = lp.add_and_infer_dtypes(
            knl,
            {"b": np.float32, "d": np.float32, "f": np.float32})
    knl = lp.prioritize_loops(knl, "i,k")
    knl = lp.prioritize_loops(knl, "i,j")

    # get a linearization
    proc_knl = preprocess_kernel(knl)
    lin_knl = get_one_linearized_kernel(proc_knl)
    linearization_items = lin_knl.linearization

    insn_id_pairs = [
        ("insn_a", "insn_b"),
        ("insn_a", "insn_c"),
        ("insn_a", "insn_d"),
        ("insn_b", "insn_c"),
        ("insn_b", "insn_d"),
        ("insn_c", "insn_d"),
        ]
    sched_maps = get_schedules_for_statement_pairs(
        proc_knl,
        linearization_items,
        insn_id_pairs,
        )

    # Relationship between insn_a and insn_b ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("insn_a", "insn_b")]

    # Create expected maps, align, compare

    sched_map_before_expected = isl.Map(
        "[pi, pk] -> { [%s=0, i, k] -> [%s] : 0 <= i < pi and 0 <= k < pk }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string(["i", "0"]),
            )
        )
    sched_map_before_expected = ensure_dim_names_match_and_align(
        sched_map_before_expected, sched_map_before)

    sched_map_after_expected = isl.Map(
        "[pi, pj] -> { [%s=1, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string(["i", "1"]),
            )
        )
    sched_map_after_expected = ensure_dim_names_match_and_align(
        sched_map_after_expected, sched_map_after)

    assert sched_map_before == sched_map_before_expected
    assert sched_map_after == sched_map_after_expected

    # ------------------------------------------------------------------------------
    # Relationship between insn_a and insn_c ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("insn_a", "insn_c")]

    # Create expected maps, align, compare

    sched_map_before_expected = isl.Map(
        "[pi, pk] -> { [%s=0, i, k] -> [%s] : 0 <= i < pi and 0 <= k < pk }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string(["i", "0"]),
            )
        )
    sched_map_before_expected = ensure_dim_names_match_and_align(
        sched_map_before_expected, sched_map_before)

    sched_map_after_expected = isl.Map(
        "[pi, pj] -> { [%s=1, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string(["i", "1"]),
            )
        )
    sched_map_after_expected = ensure_dim_names_match_and_align(
        sched_map_after_expected, sched_map_after)

    assert sched_map_before == sched_map_before_expected
    assert sched_map_after == sched_map_after_expected

    # ------------------------------------------------------------------------------
    # Relationship between insn_a and insn_d ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("insn_a", "insn_d")]

    # Create expected maps, align, compare

    sched_map_before_expected = isl.Map(
        "[pi, pk] -> { [%s=0, i, k] -> [%s] : 0 <= i < pi and 0 <= k < pk }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string([0, ]),
            )
        )
    sched_map_before_expected = ensure_dim_names_match_and_align(
        sched_map_before_expected, sched_map_before)

    sched_map_after_expected = isl.Map(
        "[pt] -> { [%s=1, t] -> [%s] : 0 <= t < pt }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string([1, ]),
            )
        )
    sched_map_after_expected = ensure_dim_names_match_and_align(
        sched_map_after_expected, sched_map_after)

    assert sched_map_before == sched_map_before_expected
    assert sched_map_after == sched_map_after_expected

    # ------------------------------------------------------------------------------
    # Relationship between insn_b and insn_c ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("insn_b", "insn_c")]

    # Create expected maps, align, compare

    sched_map_before_expected = isl.Map(
        "[pi, pj] -> { [%s=0, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string(["i", "j", 0]),
            )
        )
    sched_map_before_expected = ensure_dim_names_match_and_align(
        sched_map_before_expected, sched_map_before)

    sched_map_after_expected = isl.Map(
        "[pi, pj] -> { [%s=1, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string(["i", "j", 1]),
            )
        )
    sched_map_after_expected = ensure_dim_names_match_and_align(
        sched_map_after_expected, sched_map_after)

    assert sched_map_before == sched_map_before_expected
    assert sched_map_after == sched_map_after_expected

    # ------------------------------------------------------------------------------
    # Relationship between insn_b and insn_d ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("insn_b", "insn_d")]

    # Create expected maps, align, compare

    sched_map_before_expected = isl.Map(
        "[pi, pj] -> { [%s=0, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string([0, ]),
            )
        )
    sched_map_before_expected = ensure_dim_names_match_and_align(
        sched_map_before_expected, sched_map_before)

    sched_map_after_expected = isl.Map(
        "[pt] -> { [%s=1, t] -> [%s] : 0 <= t < pt }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string([1, ]),
            )
        )
    sched_map_after_expected = ensure_dim_names_match_and_align(
        sched_map_after_expected, sched_map_after)

    assert sched_map_before == sched_map_before_expected
    assert sched_map_after == sched_map_after_expected

    # ------------------------------------------------------------------------------
    # Relationship between insn_c and insn_d ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("insn_c", "insn_d")]

    # Create expected maps, align, compare

    sched_map_before_expected = isl.Map(
        "[pi, pj] -> { [%s=0, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string([0, ]),
            )
        )
    sched_map_before_expected = ensure_dim_names_match_and_align(
        sched_map_before_expected, sched_map_before)

    sched_map_after_expected = isl.Map(
        "[pt] -> { [%s=1, t] -> [%s] : 0 <= t < pt }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string([1, ]),
            )
        )
    sched_map_after_expected = ensure_dim_names_match_and_align(
        sched_map_after_expected, sched_map_after)

    assert sched_map_before == sched_map_before_expected
    assert sched_map_after == sched_map_after_expected


def test_pairwise_schedule_creation_parallel():
    import islpy as isl
    from loopy.schedule.checker import (
        get_schedules_for_statement_pairs,
    )
    from loopy.schedule.checker.utils import (
        ensure_dim_names_match_and_align,
    )

    # example kernel
    knl = lp.make_kernel(
        [
            "{[i]: 0<=i<pi}",
            "{[k]: 0<=k<pk}",
            "{[j,jj]: 0<=j,jj<pj}",
            "{[t]: 0<=t<pt}",
        ],
        """
        for i
            for k
                <>temp = b[i,k]  {id=insn_a}
            end
            for j
                for jj
                    a[i,j,jj] = temp + 1  {id=insn_b,dep=insn_a}
                    c[i,j,jj] = d[i,j,jj]  {id=insn_c,dep=insn_b}
                end
            end
        end
        for t
            e[t] = f[t]  {id=insn_d, dep=insn_c}
        end
        """,
        name="example",
        assumptions="pi,pj,pk,pt >= 1",
        )
    knl = lp.add_and_infer_dtypes(
            knl,
            {"b": np.float32, "d": np.float32, "f": np.float32})
    knl = lp.prioritize_loops(knl, "i,k")
    knl = lp.tag_inames(knl, {"j": "l.1", "jj": "l.0", "t": "g.0"})

    # get a linearization
    proc_knl = preprocess_kernel(knl)
    lin_knl = get_one_linearized_kernel(proc_knl)
    linearization_items = lin_knl.linearization

    insn_id_pairs = [
        ("insn_a", "insn_b"),
        ("insn_a", "insn_c"),
        ("insn_a", "insn_d"),
        ("insn_b", "insn_c"),
        ("insn_b", "insn_d"),
        ("insn_c", "insn_d"),
        ]
    sched_maps = get_schedules_for_statement_pairs(
        proc_knl,
        linearization_items,
        insn_id_pairs,
        )

    # Relationship between insn_a and insn_b ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("insn_a", "insn_b")]

    # Create expected maps, align, compare

    sched_map_before_expected = isl.Map(
        "[pi, pk] -> { [%s=0, i, k] -> [%s] : 0 <= i < pi and 0 <= k < pk }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string(["i", "0"], lid_axes=[0, 1], gid_axes=[0]),
            )
        )
    sched_map_before_expected = ensure_dim_names_match_and_align(
        sched_map_before_expected, sched_map_before)

    sched_map_after_expected = isl.Map(
        "[pi, pj] -> { [%s=1, i, j, jj] -> [%s] : 0 <= i < pi and 0 <= j,jj < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string(["i", "1"], lid_axes=[0, 1], gid_axes=[0]),
            )
        )
    sched_map_after_expected = ensure_dim_names_match_and_align(
        sched_map_after_expected, sched_map_after)

    assert sched_map_before == sched_map_before_expected
    assert sched_map_after == sched_map_after_expected

# }}}


# {{{ test lex order map creation

def test_lex_order_map_creation():
    import islpy as isl
    from loopy.schedule.checker.lexicographic_order_map import (
        create_lex_order_map,
    )
    from loopy.schedule.checker.utils import (
        append_marker_to_isl_map_var_names,
    )

    def _check_lex_map(expected_lex_order_map, n_dims):
        # Isl ignores the apostrophes, so explicitly add them
        expected_lex_order_map = append_marker_to_isl_map_var_names(
            expected_lex_order_map, isl.dim_type.in_, "'")

        lex_order_map = create_lex_order_map(
            n_dims=n_dims,
            before_names=["%s%d'" % (LEX_VAR_PREFIX, i) for i in range(n_dims)],
            after_names=["%s%d" % (LEX_VAR_PREFIX, i) for i in range(n_dims)],
            )

        assert lex_order_map == expected_lex_order_map
        assert (
            lex_order_map.get_var_names(isl.dim_type.in_) ==
            expected_lex_order_map.get_var_names(isl.dim_type.in_))
        assert (
            lex_order_map.get_var_names(isl.dim_type.out) ==
            expected_lex_order_map.get_var_names(isl.dim_type.out))

    expected_lex_order_map = isl.Map(
        "{{ "
        "[{0}0', {0}1', {0}2', {0}3', {0}4'] -> [{0}0, {0}1, {0}2, {0}3, {0}4] :"
        "("
        "{0}0' < {0}0 "
        ") or ("
        "{0}0'={0}0 and {0}1' < {0}1 "
        ") or ("
        "{0}0'={0}0 and {0}1'={0}1 and {0}2' < {0}2 "
        ") or ("
        "{0}0'={0}0 and {0}1'={0}1 and {0}2'={0}2 and {0}3' < {0}3 "
        ") or ("
        "{0}0'={0}0 and {0}1'={0}1 and {0}2'={0}2 and {0}3'={0}3 and {0}4' < {0}4"
        ")"
        "}}".format(LEX_VAR_PREFIX))

    _check_lex_map(expected_lex_order_map, 5)

    expected_lex_order_map = isl.Map(
        "{{ "
        "[{0}0'] -> [{0}0] :"
        "("
        "{0}0' < {0}0 "
        ")"
        "}}".format(LEX_VAR_PREFIX))

    _check_lex_map(expected_lex_order_map, 1)

# }}}


# {{{ test statement instance ordering creation

def test_statement_instance_ordering_creation():
    import islpy as isl
    from loopy.schedule.checker import (
        get_schedules_for_statement_pairs,
    )
    from loopy.schedule.checker.utils import (
        ensure_dim_names_match_and_align,
        append_marker_to_isl_map_var_names,
    )
    from loopy.schedule.checker.lexicographic_order_map import (
        get_statement_ordering_map,
        create_lex_order_map,
    )

    # example kernel (add deps to fix loop order)
    knl = lp.make_kernel(
        [
            "{[i]: 0<=i<pi}",
            "{[k]: 0<=k<pk}",
            "{[j]: 0<=j<pj}",
            "{[t]: 0<=t<pt}",
        ],
        """
        for i
            for k
                <>temp = b[i,k]  {id=insn_a}
            end
            for j
                a[i,j] = temp + 1  {id=insn_b,dep=insn_a}
                c[i,j] = d[i,j]  {id=insn_c,dep=insn_b}
            end
        end
        for t
            e[t] = f[t]  {id=insn_d, dep=insn_c}
        end
        """,
        name="example",
        assumptions="pi,pj,pk,pt >= 1",
        lang_version=(2018, 2)
        )
    knl = lp.add_and_infer_dtypes(
            knl,
            {"b": np.float32, "d": np.float32, "f": np.float32})
    knl = lp.prioritize_loops(knl, "i,k")
    knl = lp.prioritize_loops(knl, "i,j")

    # get a linearization
    knl = preprocess_kernel(knl)
    knl = get_one_linearized_kernel(knl)
    linearization_items = knl.linearization

    # Get pairwise schedules
    insn_id_pairs = [
        ("insn_a", "insn_b"),
        ("insn_a", "insn_c"),
        ("insn_a", "insn_d"),
        ("insn_b", "insn_c"),
        ("insn_b", "insn_d"),
        ("insn_c", "insn_d"),
        ]
    sched_maps = get_schedules_for_statement_pairs(
        knl,
        linearization_items,
        insn_id_pairs,
        )

    def check_sio_for_insn_pair(
            insn_id_before,
            insn_id_after,
            expected_lex_dims,
            expected_sio,
            ):

        # Get pairwise schedule
        (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
            (insn_id_before, insn_id_after)]

        # Get expected lex order map
        expected_lex_order_map = create_lex_order_map(
            n_dims=expected_lex_dims,
            before_names=["%s%d'" % (LEX_VAR_PREFIX, i)
                for i in range(expected_lex_dims)],
            after_names=["%s%d" % (LEX_VAR_PREFIX, i)
                for i in range(expected_lex_dims)],
            )

        assert sched_lex_order_map == expected_lex_order_map

        # create statement instance ordering,
        # maps each statement instance to all statement instances occuring later
        sio = get_statement_ordering_map(
            sched_map_before,
            sched_map_after,
            sched_lex_order_map,
            )

        sio_aligned = ensure_dim_names_match_and_align(sio, expected_sio)

        assert sio_aligned == expected_sio

    # Relationship between insn_a and insn_b ---------------------------------------

    expected_sio = isl.Map(
        "[pi, pj, pk] -> {{ "
        "[{0}'=0, i', k'] -> [{0}=1, i, j] : "
        "0 <= i' < pi and 0 <= k' < pk and 0 <= j < pj and 0 <= i < pi and i > i'; "
        "[{0}'=0, i', k'] -> [{0}=1, i=i', j] : "
        "0 <= i' < pi and 0 <= k' < pk and 0 <= j < pj "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    check_sio_for_insn_pair("insn_a", "insn_b", 2, expected_sio)

    # Relationship between insn_a and insn_c ---------------------------------------

    expected_sio = isl.Map(
        "[pi, pj, pk] -> {{ "
        "[{0}'=0, i', k'] -> [{0}=1, i, j] : "
        "0 <= i' < pi and 0 <= k' < pk and 0 <= j < pj and 0 <= i < pi and i > i'; "
        "[{0}'=0, i', k'] -> [{0}=1, i=i', j] : "
        "0 <= i' < pi and 0 <= k' < pk and 0 <= j < pj "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    check_sio_for_insn_pair("insn_a", "insn_c", 2, expected_sio)

    # Relationship between insn_a and insn_d ---------------------------------------

    expected_sio = isl.Map(
        "[pt, pi, pk] -> {{ "
        "[{0}'=0, i', k'] -> [{0}=1, t] : "
        "0 <= i' < pi and 0 <= k' < pk and 0 <= t < pt "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    check_sio_for_insn_pair("insn_a", "insn_d", 1, expected_sio)

    # Relationship between insn_b and insn_c ---------------------------------------

    expected_sio = isl.Map(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', j'] -> [{0}=1, i, j] : "
        "0 <= i' < pi and 0 <= j' < pj and i > i' and 0 <= i < pi and 0 <= j < pj; "
        "[{0}'=0, i', j'] -> [{0}=1, i=i', j] : "
        "0 <= i' < pi and 0 <= j' < pj and j > j' and 0 <= j < pj; "
        "[{0}'=0, i', j'] -> [{0}=1, i=i', j=j'] : "
        "0 <= i' < pi and 0 <= j' < pj "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    check_sio_for_insn_pair("insn_b", "insn_c", 3, expected_sio)

    # Relationship between insn_b and insn_d ---------------------------------------

    expected_sio = isl.Map(
        "[pt, pi, pj] -> {{ "
        "[{0}'=0, i', j'] -> [{0}=1, t] : "
        "0 <= i' < pi and 0 <= j' < pj and 0 <= t < pt "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    check_sio_for_insn_pair("insn_b", "insn_d", 1, expected_sio)

    # Relationship between insn_c and insn_d ---------------------------------------

    expected_sio = isl.Map(
        "[pt, pi, pj] -> {{ "
        "[{0}'=0, i', j'] -> [{0}=1, t] : "
        "0 <= i' < pi and 0 <= j' < pj and 0 <= t < pt "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    check_sio_for_insn_pair("insn_c", "insn_d", 1, expected_sio)

# TODO test SIO creation with parallel loops

# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

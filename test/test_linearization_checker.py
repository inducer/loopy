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


# {{{ helper functions for map creation/handling

def _align_and_compare_maps(maps1, maps2):
    from loopy.schedule.checker.utils import (
        ensure_dim_names_match_and_align,
    )

    for map1, map2 in zip(maps1, maps2):
        # Align maps and compare
        map1_aligned = ensure_dim_names_match_and_align(map1, map2)
        assert map1_aligned == map2


def _lex_point_string(dim_vals, lid_inames=[], gid_inames=[]):
    # Return a string describing a point in a lex space
    # by assigning values to lex dimension variables
    # (used to create maps below)

    return ", ".join(
        ["%s%d=%s" % (LEX_VAR_PREFIX, idx, str(val))
        for idx, val in enumerate(dim_vals)] +
        ["%s=%s" % (LTAG_VAR_NAMES[idx], iname)
        for idx, iname in enumerate(lid_inames)] +
        ["%s=%s" % (GTAG_VAR_NAMES[idx], iname)
        for idx, iname in enumerate(gid_inames)]
        )

# }}}


# {{{ test pairwise schedule creation

def test_pairwise_schedule_creation():
    import islpy as isl
    from loopy.schedule.checker import (
        get_schedules_for_statement_pairs,
    )

    # Example kernel
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

    # Get a linearization
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
        lin_knl,
        linearization_items,
        insn_id_pairs,
        )

    # Relationship between insn_a and insn_b ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("insn_a", "insn_b")]

    # Create expected maps and compare

    sched_map_before_expected = isl.Map(
        "[pi, pk] -> { [%s=0, i, k] -> [%s] : 0 <= i < pi and 0 <= k < pk }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["i", "0"]),
            )
        )

    sched_map_after_expected = isl.Map(
        "[pi, pj] -> { [%s=1, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["i", "1"]),
            )
        )

    _align_and_compare_maps(
        [sched_map_before_expected, sched_map_after_expected],
        [sched_map_before, sched_map_after],
        )

    # ------------------------------------------------------------------------------
    # Relationship between insn_a and insn_c ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("insn_a", "insn_c")]

    # Create expected maps and compare

    sched_map_before_expected = isl.Map(
        "[pi, pk] -> { [%s=0, i, k] -> [%s] : 0 <= i < pi and 0 <= k < pk }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["i", "0"]),
            )
        )

    sched_map_after_expected = isl.Map(
        "[pi, pj] -> { [%s=1, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["i", "1"]),
            )
        )

    _align_and_compare_maps(
        [sched_map_before_expected, sched_map_after_expected],
        [sched_map_before, sched_map_after],
        )

    # ------------------------------------------------------------------------------
    # Relationship between insn_a and insn_d ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("insn_a", "insn_d")]

    # Create expected maps and compare

    sched_map_before_expected = isl.Map(
        "[pi, pk] -> { [%s=0, i, k] -> [%s] : 0 <= i < pi and 0 <= k < pk }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string([0, ]),
            )
        )

    sched_map_after_expected = isl.Map(
        "[pt] -> { [%s=1, t] -> [%s] : 0 <= t < pt }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string([1, ]),
            )
        )

    _align_and_compare_maps(
        [sched_map_before_expected, sched_map_after_expected],
        [sched_map_before, sched_map_after],
        )

    # ------------------------------------------------------------------------------
    # Relationship between insn_b and insn_c ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("insn_b", "insn_c")]

    # Create expected maps and compare

    sched_map_before_expected = isl.Map(
        "[pi, pj] -> { [%s=0, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["i", "j", 0]),
            )
        )

    sched_map_after_expected = isl.Map(
        "[pi, pj] -> { [%s=1, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["i", "j", 1]),
            )
        )

    _align_and_compare_maps(
        [sched_map_before_expected, sched_map_after_expected],
        [sched_map_before, sched_map_after],
        )

    # ------------------------------------------------------------------------------
    # Relationship between insn_b and insn_d ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("insn_b", "insn_d")]

    # Create expected maps and compare

    sched_map_before_expected = isl.Map(
        "[pi, pj] -> { [%s=0, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string([0, ]),
            )
        )

    sched_map_after_expected = isl.Map(
        "[pt] -> { [%s=1, t] -> [%s] : 0 <= t < pt }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string([1, ]),
            )
        )

    _align_and_compare_maps(
        [sched_map_before_expected, sched_map_after_expected],
        [sched_map_before, sched_map_after],
        )

    # ------------------------------------------------------------------------------
    # Relationship between insn_c and insn_d ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("insn_c", "insn_d")]

    # Create expected maps and compare

    sched_map_before_expected = isl.Map(
        "[pi, pj] -> { [%s=0, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string([0, ]),
            )
        )

    sched_map_after_expected = isl.Map(
        "[pt] -> { [%s=1, t] -> [%s] : 0 <= t < pt }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string([1, ]),
            )
        )

    _align_and_compare_maps(
        [sched_map_before_expected, sched_map_after_expected],
        [sched_map_before, sched_map_after],
        )


def test_pairwise_schedule_creation_with_hw_par_tags():
    import islpy as isl
    from loopy.schedule.checker import (
        get_schedules_for_statement_pairs,
    )

    # Example kernel
    knl = lp.make_kernel(
        [
            "{[i,ii]: 0<=i,ii<pi}",
            "{[j,jj]: 0<=j,jj<pj}",
        ],
        """
        for i
            for ii
                for j
                    for jj
                        <>temp = b[i,ii,j,jj]  {id=stmt_a}
                        a[i,ii,j,jj] = temp + 1  {id=stmt_b,dep=stmt_a}
                    end
                end
            end
        end
        """,
        name="example",
        assumptions="pi,pj >= 1",
        lang_version=(2018, 2)
        )
    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32, "b": np.float32})
    knl = lp.tag_inames(knl, {"j": "l.1", "jj": "l.0", "i": "g.0"})

    # Get a linearization
    proc_knl = preprocess_kernel(knl)
    lin_knl = get_one_linearized_kernel(proc_knl)
    linearization_items = lin_knl.linearization

    stmt_id_pairs = [
        ("stmt_a", "stmt_b"),
        ]
    sched_maps = get_schedules_for_statement_pairs(
        lin_knl,
        linearization_items,
        stmt_id_pairs,
        )

    # Relationship between stmt_a and stmt_b ---------------------------------------

    # Get two maps
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        ("stmt_a", "stmt_b")]

    # Create expected maps and compare

    sched_map_before_expected = isl.Map(
        "[pi,pj] -> {[%s=0,i,ii,j,jj] -> [%s] : 0 <= i,ii < pi and 0 <= j,jj < pj}"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["ii", "0"], lid_inames=["jj", "j"], gid_inames=["i"]),
            )
        )

    sched_map_after_expected = isl.Map(
        "[pi,pj] -> {[%s=1,i,ii,j,jj] -> [%s] : 0 <= i,ii < pi and 0 <= j,jj < pj}"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["ii", "1"], lid_inames=["jj", "j"], gid_inames=["i"]),
            )
        )

    _align_and_compare_maps(
        [sched_map_before_expected, sched_map_after_expected],
        [sched_map_before, sched_map_after],
        )

    # ------------------------------------------------------------------------------

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

    def _check_lex_map(
            expected_lex_order_map, n_dims, lid_axes_used=[], gid_axes_used=[]):

        # Isl ignores the apostrophes, so explicitly add them
        expected_lex_order_map = append_marker_to_isl_map_var_names(
            expected_lex_order_map, isl.dim_type.in_, "'")

        lex_order_map = create_lex_order_map(
            n_dims=n_dims,
            before_names=["%s%d'" % (LEX_VAR_PREFIX, i) for i in range(n_dims)],
            after_names=["%s%d" % (LEX_VAR_PREFIX, i) for i in range(n_dims)],
            after_names_concurrent=[
                LTAG_VAR_NAMES[i] for i in lid_axes_used] + [
                GTAG_VAR_NAMES[i] for i in gid_axes_used],
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

    # Lex map for kernel with parallel HW tags

    lid_axes_used = [0, 1]
    gid_axes_used = [0, 1, 2]
    hw_par_lex_vars = [
        LTAG_VAR_NAMES[i] for i in lid_axes_used] + [
        GTAG_VAR_NAMES[i] for i in gid_axes_used]
    expected_lex_order_map = isl.Map(
        "{{ "
        "[{0}0', {0}1', {0}2', {1}', {2}', {3}', {4}', {5}'] "
        "-> [{0}0, {0}1, {0}2, {1}, {2}, {3}, {4}, {5}] :"
        "(("
        "{0}0' < {0}0 "
        ") or ("
        "{0}0'={0}0 and {0}1' < {0}1 "
        ") or ("
        "{0}0'={0}0 and {0}1'={0}1 and {0}2' < {0}2 "
        ")) and ("
        "{1}' = {1} and {2}' = {2} and {3}' = {3} and {4}' = {4} and {5}' = {5}"
        ")"
        "}}".format(LEX_VAR_PREFIX, *hw_par_lex_vars))

    _check_lex_map(
        expected_lex_order_map, 3,
        lid_axes_used=lid_axes_used, gid_axes_used=gid_axes_used)

# }}}


# {{{ test statement instance ordering creation

def _check_sio_for_stmt_pair(
        expected_sio,
        stmt_id_before,
        stmt_id_after,
        sched_maps,
        ):
    from loopy.schedule.checker.lexicographic_order_map import (
        get_statement_ordering_map,
    )
    from loopy.schedule.checker.utils import (
        ensure_dim_names_match_and_align,
    )

    # Get pairwise schedule
    (sched_map_before, sched_map_after), sched_lex_order_map = sched_maps[
        (stmt_id_before, stmt_id_after)]

    # Create statement instance ordering,
    # maps each statement instance to all statement instances occuring later
    sio = get_statement_ordering_map(
        sched_map_before,
        sched_map_after,
        sched_lex_order_map,
        )

    sio_aligned = ensure_dim_names_match_and_align(sio, expected_sio)

    assert sio_aligned == expected_sio


def test_statement_instance_ordering():
    import islpy as isl
    from loopy.schedule.checker import (
        get_schedules_for_statement_pairs,
    )
    from loopy.schedule.checker.utils import (
        append_marker_to_isl_map_var_names,
    )

    # Example kernel (add deps to fix loop order)
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
                <>temp = b[i,k]  {id=stmt_a}
            end
            for j
                a[i,j] = temp + 1  {id=stmt_b,dep=stmt_a}
                c[i,j] = d[i,j]  {id=stmt_c,dep=stmt_b}
            end
        end
        for t
            e[t] = f[t]  {id=stmt_d, dep=stmt_c}
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

    # Get a linearization
    knl = preprocess_kernel(knl)
    knl = get_one_linearized_kernel(knl)
    linearization_items = knl.linearization

    # Get pairwise schedules
    stmt_id_pairs = [
        ("stmt_a", "stmt_b"),
        ("stmt_a", "stmt_c"),
        ("stmt_a", "stmt_d"),
        ("stmt_b", "stmt_c"),
        ("stmt_b", "stmt_d"),
        ("stmt_c", "stmt_d"),
        ]
    sched_maps = get_schedules_for_statement_pairs(
        knl,
        linearization_items,
        stmt_id_pairs,
        )

    # Relationship between stmt_a and stmt_b ---------------------------------------

    expected_sio = isl.Map(
        "[pi, pj, pk] -> {{ "
        "[{0}'=0, i', k'] -> [{0}=1, i, j] : "
        "0 <= i,i' < pi and 0 <= k' < pk and 0 <= j < pj and i >= i' "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    _check_sio_for_stmt_pair(expected_sio, "stmt_a", "stmt_b", sched_maps)

    # Relationship between stmt_a and stmt_c ---------------------------------------

    expected_sio = isl.Map(
        "[pi, pj, pk] -> {{ "
        "[{0}'=0, i', k'] -> [{0}=1, i, j] : "
        "0 <= i,i' < pi and 0 <= k' < pk and 0 <= j < pj and i >= i' "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    _check_sio_for_stmt_pair(expected_sio, "stmt_a", "stmt_c", sched_maps)

    # Relationship between stmt_a and stmt_d ---------------------------------------

    expected_sio = isl.Map(
        "[pt, pi, pk] -> {{ "
        "[{0}'=0, i', k'] -> [{0}=1, t] : "
        "0 <= i' < pi and 0 <= k' < pk and 0 <= t < pt "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    _check_sio_for_stmt_pair(expected_sio, "stmt_a", "stmt_d", sched_maps)

    # Relationship between stmt_b and stmt_c ---------------------------------------

    expected_sio = isl.Map(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', j'] -> [{0}=1, i, j] : "
        "0 <= i,i' < pi and 0 <= j,j' < pj and i > i'; "
        "[{0}'=0, i', j'] -> [{0}=1, i=i', j] : "
        "0 <= i' < pi and 0 <= j,j' < pj and j >= j'; "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    _check_sio_for_stmt_pair(expected_sio, "stmt_b", "stmt_c", sched_maps)

    # Relationship between stmt_b and stmt_d ---------------------------------------

    expected_sio = isl.Map(
        "[pt, pi, pj] -> {{ "
        "[{0}'=0, i', j'] -> [{0}=1, t] : "
        "0 <= i' < pi and 0 <= j' < pj and 0 <= t < pt "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    _check_sio_for_stmt_pair(expected_sio, "stmt_b", "stmt_d", sched_maps)

    # Relationship between stmt_c and stmt_d ---------------------------------------

    expected_sio = isl.Map(
        "[pt, pi, pj] -> {{ "
        "[{0}'=0, i', j'] -> [{0}=1, t] : "
        "0 <= i' < pi and 0 <= j' < pj and 0 <= t < pt "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    _check_sio_for_stmt_pair(expected_sio, "stmt_c", "stmt_d", sched_maps)


def test_statement_instance_ordering_with_hw_par_tags():
    import islpy as isl
    from loopy.schedule.checker import (
        get_schedules_for_statement_pairs,
    )
    from loopy.schedule.checker.utils import (
        append_marker_to_isl_map_var_names,
        partition_inames_by_concurrency,
    )

    # Example kernel
    knl = lp.make_kernel(
        [
            "{[i,ii]: 0<=i,ii<pi}",
            "{[j,jj]: 0<=j,jj<pj}",
        ],
        """
        for i
            for ii
                for j
                    for jj
                        <>temp = b[i,ii,j,jj]  {id=stmt_a}
                        a[i,ii,j,jj] = temp + 1  {id=stmt_b,dep=stmt_a}
                    end
                end
            end
        end
        """,
        name="example",
        assumptions="pi,pj >= 1",
        lang_version=(2018, 2)
        )
    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32, "b": np.float32})
    knl = lp.tag_inames(knl, {"j": "l.1", "jj": "l.0", "i": "g.0"})

    # Get a linearization
    proc_knl = preprocess_kernel(knl)
    lin_knl = get_one_linearized_kernel(proc_knl)
    linearization_items = lin_knl.linearization

    # Get pairwise schedules
    stmt_id_pairs = [
        ("stmt_a", "stmt_b"),
        ]
    sched_maps = get_schedules_for_statement_pairs(
        lin_knl,
        linearization_items,
        stmt_id_pairs,
        )

    # Create string for representing parallel iname condition in sio
    conc_inames, _ = partition_inames_by_concurrency(knl)
    par_iname_condition = " and ".join(
        "{0} = {0}'".format(iname) for iname in conc_inames)

    # Relationship between stmt_a and stmt_b ---------------------------------------

    expected_sio = isl.Map(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii', j', jj'] -> [{0}=1, i, ii, j, jj] : "
        "0 <= i,ii,i',ii' < pi and 0 <= j,jj,j',jj' < pj and ii >= ii' "
        "and {1} "
        "}}".format(
            STATEMENT_VAR_NAME,
            par_iname_condition,
            )
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    _check_sio_for_stmt_pair(expected_sio, "stmt_a", "stmt_b", sched_maps)

    # ------------------------------------------------------------------------------

# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

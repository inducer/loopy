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
import islpy as isl
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
    BEFORE_MARK,
)
from loopy.schedule.checker.utils import (
    ensure_dim_names_match_and_align,
)

logger = logging.getLogger(__name__)


# {{{ Helper functions for map creation/handling

def _align_and_compare_maps(maps):
    from loopy.schedule.checker.utils import prettier_map_string

    for map1, map2 in maps:
        # Align maps and compare
        map1_aligned = ensure_dim_names_match_and_align(map1, map2)
        if map1_aligned != map2:
            print("Maps not equal:")
            print(prettier_map_string(map1_aligned))
            print(prettier_map_string(map2))
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


def _isl_map_with_marked_dims(s, placeholder_mark="'"):
    # For creating legible tests, map strings may be created with a placeholder
    # for the 'before' mark. Replace this placeholder with BEFORE_MARK before
    # creating the map.
    # ALSO, if BEFORE_MARK == "'", ISL will ignore this mark when creating
    # variable names, so it must be added manually.
    from loopy.schedule.checker.utils import (
        append_mark_to_isl_map_var_names,
    )
    dt = isl.dim_type
    if BEFORE_MARK == "'":
        # ISL will ignore the apostrophe; manually name the in_ vars
        return append_mark_to_isl_map_var_names(
            isl.Map(s.replace(placeholder_mark, BEFORE_MARK)),
            dt.in_,
            BEFORE_MARK)
    else:
        return isl.Map(s.replace(placeholder_mark, BEFORE_MARK))


def _check_orderings_for_stmt_pair(
        stmt_id_before,
        stmt_id_after,
        all_sios,
        sio_intra_thread_exp=None,
        sched_before_intra_thread_exp=None,
        sched_after_intra_thread_exp=None,
        sio_intra_group_exp=None,
        sched_before_intra_group_exp=None,
        sched_after_intra_group_exp=None,
        sio_global_exp=None,
        sched_before_global_exp=None,
        sched_after_global_exp=None,
        ):

    order_info = all_sios[(stmt_id_before, stmt_id_after)]

    # Get pairs of maps to compare for equality
    map_candidates = zip([
        sio_intra_thread_exp,
        sched_before_intra_thread_exp, sched_after_intra_thread_exp,
        sio_intra_group_exp,
        sched_before_intra_group_exp, sched_after_intra_group_exp,
        sio_global_exp,
        sched_before_global_exp, sched_after_global_exp,
        ], [
        order_info.sio_intra_thread,
        order_info.pwsched_intra_thread[0], order_info.pwsched_intra_thread[1],
        order_info.sio_intra_group,
        order_info.pwsched_intra_group[0], order_info.pwsched_intra_group[1],
        order_info.sio_global,
        order_info.pwsched_global[0], order_info.pwsched_global[1],
        ])

    # Only compare to maps that were passed
    maps_to_compare = [(m1, m2) for m1, m2 in map_candidates if m1 is not None]
    _align_and_compare_maps(maps_to_compare)


def _process_and_linearize(knl):
    # Return linearization items along with the preprocessed kernel and
    # linearized kernel
    proc_knl = preprocess_kernel(knl)
    lin_knl = get_one_linearized_kernel(proc_knl)
    return lin_knl.linearization, proc_knl, lin_knl


def _get_runinstruction_ids_from_linearization(lin_items):
    from loopy.schedule import RunInstruction
    return [
        lin_item.insn_id for lin_item in lin_items
        if isinstance(lin_item, RunInstruction)]

# }}}


# {{{ test_intra_thread_pairwise_schedule_creation()

def test_intra_thread_pairwise_schedule_creation():
    from loopy.schedule.checker import (
        get_pairwise_statement_orderings,
    )

    # Example kernel
    # stmt_c depends on stmt_b only to create deterministic order
    # stmt_d depends on stmt_c only to create deterministic order
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
        )
    knl = lp.add_and_infer_dtypes(
            knl,
            {"b": np.float32, "d": np.float32, "f": np.float32})
    knl = lp.prioritize_loops(knl, "i,k")
    knl = lp.prioritize_loops(knl, "i,j")

    # Get a linearization
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    stmt_id_pairs = [
        ("stmt_a", "stmt_b"),
        ("stmt_a", "stmt_c"),
        ("stmt_a", "stmt_d"),
        ("stmt_b", "stmt_c"),
        ("stmt_b", "stmt_d"),
        ("stmt_c", "stmt_d"),
        ]
    pworders = get_pairwise_statement_orderings(
        lin_knl,
        lin_items,
        stmt_id_pairs,
        )

    # {{{ Relationship between stmt_a and stmt_b

    # Create expected maps and compare

    sched_stmt_a_intra_thread_exp = isl.Map(
        "[pi, pk] -> { [%s=0, i, k] -> [%s] : 0 <= i < pi and 0 <= k < pk }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["i", "0"]),
            )
        )

    sched_stmt_b_intra_thread_exp = isl.Map(
        "[pi, pj] -> { [%s=1, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["i", "1"]),
            )
        )

    _check_orderings_for_stmt_pair(
        "stmt_a", "stmt_b", pworders,
        sched_before_intra_thread_exp=sched_stmt_a_intra_thread_exp,
        sched_after_intra_thread_exp=sched_stmt_b_intra_thread_exp,
        )

    # }}}

    # {{{ Relationship between stmt_a and stmt_c

    # Create expected maps and compare

    sched_stmt_a_intra_thread_exp = isl.Map(
        "[pi, pk] -> { [%s=0, i, k] -> [%s] : 0 <= i < pi and 0 <= k < pk }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["i", "0"]),
            )
        )

    sched_stmt_c_intra_thread_exp = isl.Map(
        "[pi, pj] -> { [%s=1, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["i", "1"]),
            )
        )

    _check_orderings_for_stmt_pair(
        "stmt_a", "stmt_c", pworders,
        sched_before_intra_thread_exp=sched_stmt_a_intra_thread_exp,
        sched_after_intra_thread_exp=sched_stmt_c_intra_thread_exp,
        )

    # }}}

    # {{{ Relationship between stmt_a and stmt_d

    # Create expected maps and compare

    sched_stmt_a_intra_thread_exp = isl.Map(
        "[pi, pk] -> { [%s=0, i, k] -> [%s] : 0 <= i < pi and 0 <= k < pk }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string([0, ]),
            )
        )

    sched_stmt_d_intra_thread_exp = isl.Map(
        "[pt] -> { [%s=1, t] -> [%s] : 0 <= t < pt }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string([1, ]),
            )
        )

    _check_orderings_for_stmt_pair(
        "stmt_a", "stmt_d", pworders,
        sched_before_intra_thread_exp=sched_stmt_a_intra_thread_exp,
        sched_after_intra_thread_exp=sched_stmt_d_intra_thread_exp,
        )

    # }}}

    # {{{ Relationship between stmt_b and stmt_c

    # Create expected maps and compare

    sched_stmt_b_intra_thread_exp = isl.Map(
        "[pi, pj] -> { [%s=0, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["i", "j", 0]),
            )
        )

    sched_stmt_c_intra_thread_exp = isl.Map(
        "[pi, pj] -> { [%s=1, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(["i", "j", 1]),
            )
        )

    _check_orderings_for_stmt_pair(
        "stmt_b", "stmt_c", pworders,
        sched_before_intra_thread_exp=sched_stmt_b_intra_thread_exp,
        sched_after_intra_thread_exp=sched_stmt_c_intra_thread_exp,
        )

    # }}}

    # {{{ Relationship between stmt_b and stmt_d

    # Create expected maps and compare

    sched_stmt_b_intra_thread_exp = isl.Map(
        "[pi, pj] -> { [%s=0, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string([0, ]),
            )
        )

    sched_stmt_d_intra_thread_exp = isl.Map(
        "[pt] -> { [%s=1, t] -> [%s] : 0 <= t < pt }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string([1, ]),
            )
        )

    _check_orderings_for_stmt_pair(
        "stmt_b", "stmt_d", pworders,
        sched_before_intra_thread_exp=sched_stmt_b_intra_thread_exp,
        sched_after_intra_thread_exp=sched_stmt_d_intra_thread_exp,
        )

    # }}}

    # {{{ Relationship between stmt_c and stmt_d

    # Create expected maps and compare

    sched_stmt_c_intra_thread_exp = isl.Map(
        "[pi, pj] -> { [%s=0, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string([0, ]),
            )
        )

    sched_stmt_d_intra_thread_exp = isl.Map(
        "[pt] -> { [%s=1, t] -> [%s] : 0 <= t < pt }"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string([1, ]),
            )
        )

    _check_orderings_for_stmt_pair(
        "stmt_c", "stmt_d", pworders,
        sched_before_intra_thread_exp=sched_stmt_c_intra_thread_exp,
        sched_after_intra_thread_exp=sched_stmt_d_intra_thread_exp,
        )

    # }}}

# }}}


# {{{ test_pairwise_schedule_creation_with_hw_par_tags()

def test_pairwise_schedule_creation_with_hw_par_tags():
    # (further sched testing in SIO tests below)

    from loopy.schedule.checker import (
        get_pairwise_statement_orderings,
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
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    stmt_id_pairs = [
        ("stmt_a", "stmt_b"),
        ]
    pworders = get_pairwise_statement_orderings(
        lin_knl,
        lin_items,
        stmt_id_pairs,
        )

    # {{{ Relationship between stmt_a and stmt_b

    # Create expected maps and compare

    sched_stmt_a_intra_thread_exp = isl.Map(
        "[pi,pj] -> {[%s=0,i,ii,j,jj] -> [%s] : 0 <= i,ii < pi and 0 <= j,jj < pj}"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["ii", "0"],
                lid_inames=["jj", "j"], gid_inames=["i"],
                ),
            )
        )

    sched_stmt_b_intra_thread_exp = isl.Map(
        "[pi,pj] -> {[%s=1,i,ii,j,jj] -> [%s] : 0 <= i,ii < pi and 0 <= j,jj < pj}"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["ii", "1"],
                lid_inames=["jj", "j"], gid_inames=["i"],
                ),
            )
        )

    _check_orderings_for_stmt_pair(
        "stmt_a", "stmt_b", pworders,
        sched_before_intra_thread_exp=sched_stmt_a_intra_thread_exp,
        sched_after_intra_thread_exp=sched_stmt_b_intra_thread_exp,
        )

    # }}}

# }}}


# {{{ test_lex_order_map_creation()

def test_lex_order_map_creation():
    from loopy.schedule.checker.lexicographic_order_map import (
        create_lex_order_map,
    )

    def _check_lex_map(exp_lex_order_map, n_dims):

        lex_order_map = create_lex_order_map(
            dim_names=["%s%d" % (LEX_VAR_PREFIX, i) for i in range(n_dims)],
            in_dim_mark=BEFORE_MARK,
            )

        assert lex_order_map == exp_lex_order_map
        assert lex_order_map.get_var_dict() == exp_lex_order_map.get_var_dict()

    exp_lex_order_map = _isl_map_with_marked_dims(
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

    _check_lex_map(exp_lex_order_map, 5)

    exp_lex_order_map = _isl_map_with_marked_dims(
        "{{ "
        "[{0}0'] -> [{0}0] :"
        "("
        "{0}0' < {0}0 "
        ")"
        "}}".format(LEX_VAR_PREFIX))

    _check_lex_map(exp_lex_order_map, 1)

# }}}


# {{{ test_intra_thread_statement_instance_ordering()

def test_intra_thread_statement_instance_ordering():
    from loopy.schedule.checker import (
        get_pairwise_statement_orderings,
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
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    # Get pairwise schedules
    stmt_id_pairs = [
        ("stmt_a", "stmt_b"),
        ("stmt_a", "stmt_c"),
        ("stmt_a", "stmt_d"),
        ("stmt_b", "stmt_c"),
        ("stmt_b", "stmt_d"),
        ("stmt_c", "stmt_d"),
        ]
    pworders = get_pairwise_statement_orderings(
        proc_knl,
        lin_items,
        stmt_id_pairs,
        )

    # {{{ Relationship between stmt_a and stmt_b

    sio_intra_thread_exp = _isl_map_with_marked_dims(
        "[pi, pj, pk] -> {{ "
        "[{0}'=0, i', k'] -> [{0}=1, i, j] : "
        "0 <= i,i' < pi and 0 <= k' < pk and 0 <= j < pj and i >= i' "
        "}}".format(STATEMENT_VAR_NAME)
        )

    _check_orderings_for_stmt_pair(
        "stmt_a", "stmt_b", pworders, sio_intra_thread_exp=sio_intra_thread_exp)

    # }}}

    # {{{ Relationship between stmt_a and stmt_c

    sio_intra_thread_exp = _isl_map_with_marked_dims(
        "[pi, pj, pk] -> {{ "
        "[{0}'=0, i', k'] -> [{0}=1, i, j] : "
        "0 <= i,i' < pi and 0 <= k' < pk and 0 <= j < pj and i >= i' "
        "}}".format(STATEMENT_VAR_NAME)
        )

    _check_orderings_for_stmt_pair(
        "stmt_a", "stmt_c", pworders, sio_intra_thread_exp=sio_intra_thread_exp)

    # }}}

    # {{{ Relationship between stmt_a and stmt_d

    sio_intra_thread_exp = _isl_map_with_marked_dims(
        "[pt, pi, pk] -> {{ "
        "[{0}'=0, i', k'] -> [{0}=1, t] : "
        "0 <= i' < pi and 0 <= k' < pk and 0 <= t < pt "
        "}}".format(STATEMENT_VAR_NAME)
        )

    _check_orderings_for_stmt_pair(
        "stmt_a", "stmt_d", pworders, sio_intra_thread_exp=sio_intra_thread_exp)

    # }}}

    # {{{ Relationship between stmt_b and stmt_c

    sio_intra_thread_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', j'] -> [{0}=1, i, j] : "
        "0 <= i,i' < pi and 0 <= j,j' < pj and i > i'; "
        "[{0}'=0, i', j'] -> [{0}=1, i=i', j] : "
        "0 <= i' < pi and 0 <= j,j' < pj and j >= j'; "
        "}}".format(STATEMENT_VAR_NAME)
        )

    _check_orderings_for_stmt_pair(
        "stmt_b", "stmt_c", pworders, sio_intra_thread_exp=sio_intra_thread_exp)

    # }}}

    # {{{ Relationship between stmt_b and stmt_d

    sio_intra_thread_exp = _isl_map_with_marked_dims(
        "[pt, pi, pj] -> {{ "
        "[{0}'=0, i', j'] -> [{0}=1, t] : "
        "0 <= i' < pi and 0 <= j' < pj and 0 <= t < pt "
        "}}".format(STATEMENT_VAR_NAME)
        )

    _check_orderings_for_stmt_pair(
        "stmt_b", "stmt_d", pworders, sio_intra_thread_exp=sio_intra_thread_exp)

    # }}}

    # {{{ Relationship between stmt_c and stmt_d

    sio_intra_thread_exp = _isl_map_with_marked_dims(
        "[pt, pi, pj] -> {{ "
        "[{0}'=0, i', j'] -> [{0}=1, t] : "
        "0 <= i' < pi and 0 <= j' < pj and 0 <= t < pt "
        "}}".format(STATEMENT_VAR_NAME)
        )

    _check_orderings_for_stmt_pair(
        "stmt_c", "stmt_d", pworders, sio_intra_thread_exp=sio_intra_thread_exp)

    # }}}

# }}}


# {{{ test_statement_instance_ordering_with_hw_par_tags()

def test_statement_instance_ordering_with_hw_par_tags():
    from loopy.schedule.checker import (
        get_pairwise_statement_orderings,
    )
    from loopy.schedule.checker.utils import (
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
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    # Get pairwise schedules
    stmt_id_pairs = [
        ("stmt_a", "stmt_b"),
        ]
    pworders = get_pairwise_statement_orderings(
        lin_knl,
        lin_items,
        stmt_id_pairs,
        )

    # Create string for representing parallel iname condition in sio
    conc_inames, _ = partition_inames_by_concurrency(knl)
    par_iname_condition = " and ".join(
        "{0} = {0}'".format(iname) for iname in conc_inames)

    # {{{ Relationship between stmt_a and stmt_b

    sio_intra_thread_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii', j', jj'] -> [{0}=1, i, ii, j, jj] : "
        "0 <= i,ii,i',ii' < pi and 0 <= j,jj,j',jj' < pj and ii >= ii' "
        "and {1} "
        "}}".format(
            STATEMENT_VAR_NAME,
            par_iname_condition,
            )
        )

    _check_orderings_for_stmt_pair(
        "stmt_a", "stmt_b", pworders, sio_intra_thread_exp=sio_intra_thread_exp)

    # }}}

# }}}


# {{{ test_sios_and_schedules_with_barriers()

def test_sios_and_schedules_with_barriers():
    from loopy.schedule.checker import (
        get_pairwise_statement_orderings,
    )

    assumptions = "ij_end >= ij_start + 1 and lg_end >= 1"
    knl = lp.make_kernel(
        [
            "{[i,j]: ij_start<=i,j<ij_end}",
            "{[l0,l1,g0]: 0<=l0,l1,g0<lg_end}",
        ],
        """
        for g0
            for l0
                for l1
                    <>temp0 = 0  {id=stmt_0}
                    ... lbarrier  {id=stmt_b0,dep=stmt_0}
                    <>temp1 = 1  {id=stmt_1,dep=stmt_b0}
                    for i
                        <>tempi0 = 0  {id=stmt_i0,dep=stmt_1}
                        ... lbarrier {id=stmt_ib0,dep=stmt_i0}
                        ... gbarrier {id=stmt_ibb0,dep=stmt_i0}
                        <>tempi1 = 0  {id=stmt_i1,dep=stmt_ib0}
                        <>tempi2 = 0  {id=stmt_i2,dep=stmt_i1}
                        for j
                            <>tempj0 = 0  {id=stmt_j0,dep=stmt_i2}
                            ... lbarrier {id=stmt_jb0,dep=stmt_j0}
                            <>tempj1 = 0  {id=stmt_j1,dep=stmt_jb0}
                        end
                    end
                    <>temp2 = 0  {id=stmt_2,dep=stmt_i0}
                end
            end
        end
        """,
        name="funky",
        assumptions=assumptions,
        lang_version=(2018, 2)
        )
    knl = lp.tag_inames(knl, {"l0": "l.0", "l1": "l.1", "g0": "g.0"})

    # Get a linearization
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    stmt_id_pairs = [("stmt_j1", "stmt_2"), ("stmt_1", "stmt_i0")]
    pworders = get_pairwise_statement_orderings(
        lin_knl, lin_items, stmt_id_pairs)

    # {{{ Relationship between stmt_j1 and stmt_2

    # Create expected maps and compare

    # Iname bound strings to facilitate creation of expected maps
    i_bound_str = "ij_start <= i < ij_end"
    i_bound_str_p = "ij_start <= i' < ij_end"
    j_bound_str = "ij_start <= j < ij_end"
    j_bound_str_p = "ij_start <= j' < ij_end"
    ij_bound_str = i_bound_str + " and " + j_bound_str
    ij_bound_str_p = i_bound_str_p + " and " + j_bound_str_p
    conc_iname_bound_str = "0 <= l0,l1,g0 < lg_end"
    conc_iname_bound_str_p = "0 <= l0',l1',g0' < lg_end"

    # {{{ Intra-group

    sched_stmt_j1_intra_group_exp = isl.Map(
        "[ij_start, ij_end, lg_end] -> {"
        "[%s=0, i, j, l0, l1, g0] -> [%s] : "
        "%s and %s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["2", "i", "2", "j", "1"],  # lex points
                lid_inames=["l0", "l1"], gid_inames=["g0"],
                ),
            ij_bound_str,
            conc_iname_bound_str,
            )
        )

    sched_stmt_2_intra_group_exp = isl.Map(
        "[lg_end] -> {[%s=1, l0, l1, g0] -> [%s] : %s}"
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["3", "0", "0", "0", "0"],  # lex points
                lid_inames=["l0", "l1"], gid_inames=["g0"],
                ),
            conc_iname_bound_str,
            )
        )

    sio_intra_group_exp = _isl_map_with_marked_dims(
        "[ij_start, ij_end, lg_end] -> {{ "
        "[{0}'=0, i', j', l0', l1', g0'] -> [{0}=1, l0, l1, g0] : "
        "(ij_start <= j' < ij_end-1 or "  # not last iteration of j
        " ij_start <= i' < ij_end-1) "  # not last iteration of i
        "and g0 = g0' "  # within a single group
        "and {1} and {2} and {3} "  # iname bounds
        "and {4}"  # param assumptions
        "}}".format(
            STATEMENT_VAR_NAME,
            ij_bound_str_p,
            conc_iname_bound_str,
            conc_iname_bound_str_p,
            assumptions,
            )
        )

    # }}}

    # {{{ Global

    sched_stmt_j1_global_exp = isl.Map(
        "[ij_start, ij_end, lg_end] -> {"
        "[%s=0, i, j, l0, l1, g0] -> [%s] : "
        "%s and %s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["1", "i", "1"],  # lex points
                lid_inames=["l0", "l1"], gid_inames=["g0"],
                ),
            ij_bound_str,
            conc_iname_bound_str,
            )
        )

    sched_stmt_2_global_exp = isl.Map(
        "[lg_end] -> {[%s=1, l0, l1, g0] -> [%s] : "
        "%s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["2", "0", "0"],  # lex points
                lid_inames=["l0", "l1"], gid_inames=["g0"],
                ),
            conc_iname_bound_str,
            )
        )

    sio_global_exp = _isl_map_with_marked_dims(
        "[ij_start,ij_end,lg_end] -> {{ "
        "[{0}'=0, i', j', l0', l1', g0'] -> [{0}=1, l0, l1, g0] : "
        "ij_start <= i' < ij_end-1 "  # not last iteration of i
        "and {1} and {2} and {3} "  # iname bounds
        "and {4}"  # param assumptions
        "}}".format(
            STATEMENT_VAR_NAME,
            ij_bound_str_p,
            conc_iname_bound_str,
            conc_iname_bound_str_p,
            assumptions,
            )
        )

    # }}}

    _check_orderings_for_stmt_pair(
        "stmt_j1", "stmt_2", pworders,
        sio_intra_group_exp=sio_intra_group_exp,
        sched_before_intra_group_exp=sched_stmt_j1_intra_group_exp,
        sched_after_intra_group_exp=sched_stmt_2_intra_group_exp,
        sio_global_exp=sio_global_exp,
        sched_before_global_exp=sched_stmt_j1_global_exp,
        sched_after_global_exp=sched_stmt_2_global_exp,
        )

    # {{{ Check for some key example pairs in the sio_intra_group map

    # Get maps
    order_info = pworders[("stmt_j1", "stmt_2")]

    # As long as this is not the last iteration of the i loop, then there
    # should be a barrier between the last instance of statement stmt_j1
    # and statement stmt_2:
    ij_end_val = 7
    last_i_val = ij_end_val - 1
    max_non_last_i_val = last_i_val - 1  # max i val that isn't the last iteration

    wanted_pairs = _isl_map_with_marked_dims(
        "[ij_start, ij_end, lg_end] -> {{"
        "[{0}' = 0, i', j'=ij_end-1, g0', l0', l1'] -> [{0} = 1, l0, l1, g0] : "
        "ij_start <= i' <= {1} "  # constrain i
        "and ij_end >= {2} "  # constrain ij_end
        "and g0 = g0' "  # within a single group
        "and {3} and {4} "  # conc iname bounds
        "}}".format(
            STATEMENT_VAR_NAME,
            max_non_last_i_val,
            ij_end_val,
            conc_iname_bound_str,
            conc_iname_bound_str_p,
            )
        )
    wanted_pairs = ensure_dim_names_match_and_align(
        wanted_pairs, order_info.sio_intra_group)

    assert wanted_pairs.is_subset(order_info.sio_intra_group)

    # If this IS the last iteration of the i loop, then there
    # should NOT be a barrier between the last instance of statement stmt_j1
    # and statement stmt_2:
    unwanted_pairs = _isl_map_with_marked_dims(
        "[ij_start, ij_end, lg_end] -> {{"
        "[{0}' = 0, i', j'=ij_end-1, g0', l0', l1'] -> [{0} = 1, l0, l1, g0] : "
        "ij_start <= i' <= {1} "  # constrain i
        "and ij_end >= {2} "  # constrain p
        "and g0 = g0' "  # within a single group
        "and {3} and {4} "  # conc iname bounds
        "}}".format(
            STATEMENT_VAR_NAME,
            last_i_val,
            ij_end_val,
            conc_iname_bound_str,
            conc_iname_bound_str_p,
            )
        )
    unwanted_pairs = ensure_dim_names_match_and_align(
        unwanted_pairs, order_info.sio_intra_group)

    assert not unwanted_pairs.is_subset(order_info.sio_intra_group)

    # }}}

    # }}}

    # {{{ Relationship between stmt_1 and stmt_i0

    # Create expected maps and compare

    # {{{ Intra-group

    sched_stmt_1_intra_group_exp = isl.Map(
        "[lg_end] -> {[%s=0, l0, l1, g0] -> [%s] : "
        "%s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["1", "0", "0", "0", "0"],  # lex points
                lid_inames=["l0", "l1"], gid_inames=["g0"],
                ),
            conc_iname_bound_str,
            )
        )

    sched_stmt_i0_intra_group_exp = isl.Map(
        "[ij_start, ij_end, lg_end] -> {"
        "[%s=1, i, l0, l1, g0] -> [%s] : "
        "%s and %s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["2", "i", "0", "0", "0"],  # lex points
                lid_inames=["l0", "l1"], gid_inames=["g0"],
                ),
            i_bound_str,
            conc_iname_bound_str,
            )
        )

    sio_intra_group_exp = _isl_map_with_marked_dims(
        "[ij_start, ij_end, lg_end] -> {{ "
        "[{0}'=0, l0', l1', g0'] -> [{0}=1, i, l0, l1, g0] : "
        "ij_start + 1 <= i < ij_end "  # not first iteration of i
        "and g0 = g0' "  # within a single group
        "and {1} and {2} and {3} "  # iname bounds
        "and {4}"  # param assumptions
        "}}".format(
            STATEMENT_VAR_NAME,
            i_bound_str,
            conc_iname_bound_str,
            conc_iname_bound_str_p,
            assumptions,
            )
        )

    # }}}

    # {{{ Global

    sched_stmt_1_global_exp = isl.Map(
        "[lg_end] -> {[%s=0, l0, l1, g0] -> [%s] : "
        "%s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["0", "0", "0"],  # lex points
                lid_inames=["l0", "l1"], gid_inames=["g0"],
                ),
            conc_iname_bound_str,
            )
        )

    sched_stmt_i0_global_exp = isl.Map(
        "[ij_start, ij_end, lg_end] -> {"
        "[%s=1, i, l0, l1, g0] -> [%s] : "
        "%s and %s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["1", "i", "0"],  # lex points
                lid_inames=["l0", "l1"], gid_inames=["g0"],
                ),
            i_bound_str,
            conc_iname_bound_str,
            )
        )

    sio_global_exp = _isl_map_with_marked_dims(
        "[ij_start, ij_end, lg_end] -> {{ "
        "[{0}'=0, l0', l1', g0'] -> [{0}=1, i, l0, l1, g0] : "
        "ij_start + 1 <= i < ij_end "  # not first iteration of i
        "and {1} and {2} and {3} "  # iname bounds
        "and {4}"  # param assumptions
        "}}".format(
            STATEMENT_VAR_NAME,
            i_bound_str,
            conc_iname_bound_str,
            conc_iname_bound_str_p,
            assumptions,
            )
        )

    # }}}

    _check_orderings_for_stmt_pair(
        "stmt_1", "stmt_i0", pworders,
        sio_intra_group_exp=sio_intra_group_exp,
        sched_before_intra_group_exp=sched_stmt_1_intra_group_exp,
        sched_after_intra_group_exp=sched_stmt_i0_intra_group_exp,
        sio_global_exp=sio_global_exp,
        sched_before_global_exp=sched_stmt_1_global_exp,
        sched_after_global_exp=sched_stmt_i0_global_exp,
        )

    # }}}

# }}}


# {{{ test_sios_and_schedules_with_vec_and_barriers()

def test_sios_and_schedules_with_vec_and_barriers():
    from loopy.schedule.checker import (
        get_pairwise_statement_orderings,
    )

    knl = lp.make_kernel(
        "{[i, j, l0] : 0 <= i < 4 and 0 <= j < n and 0 <= l0 < 32}",
        """
        for l0
            for i
                for j
                    b[i,j,l0] = 1 {id=stmt_1}
                    ... lbarrier  {id=b,dep=stmt_1}
                    c[i,j,l0] = 2 {id=stmt_2, dep=b}
                end
            end
        end
        """)
    knl = lp.add_and_infer_dtypes(knl, {"b": "float32", "c": "float32"})

    knl = lp.tag_inames(knl, {"i": "vec", "l0": "l.0"})

    # Get a linearization
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    stmt_id_pairs = [("stmt_1", "stmt_2")]
    pworders = get_pairwise_statement_orderings(
        lin_knl, lin_items, stmt_id_pairs)

    # {{{ Relationship between stmt_1 and stmt_2

    # Create expected maps and compare

    # Iname bound strings to facilitate creation of expected maps
    ij_bound_str = "0 <= i < 4 and 0 <= j < n"
    ij_bound_str_p = "0 <= i' < 4 and 0 <= j' < n"
    conc_iname_bound_str = "0 <= l0 < 32"
    conc_iname_bound_str_p = "0 <= l0' < 32"

    # {{{ Intra-thread

    sched_stmt_1_intra_thread_exp = isl.Map(
        "[n] -> {"
        "[%s=0, i, j, l0] -> [%s] : "
        "%s and %s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["j", "0"],  # lex points (initial matching dim gets removed)
                lid_inames=["l0"],
                ),
            ij_bound_str,
            conc_iname_bound_str,
            )
        )

    sched_stmt_2_intra_thread_exp = isl.Map(
        "[n] -> {"
        "[%s=1, i, j, l0] -> [%s] : "
        "%s and %s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["j", "1"],  # lex points (initial matching dim gets removed)
                lid_inames=["l0"],
                ),
            ij_bound_str,
            conc_iname_bound_str,
            )
        )

    sio_intra_thread_exp = _isl_map_with_marked_dims(
        "[n] -> {{ "
        "[{0}'=0, i', j', l0'] -> [{0}=1, i, j, l0] : "
        "j' <= j "
        "and l0 = l0' "  # within a single thread
        "and {1} and {2} and {3} and {4}"  # iname bounds
        "}}".format(
            STATEMENT_VAR_NAME,
            ij_bound_str,
            ij_bound_str_p,
            conc_iname_bound_str,
            conc_iname_bound_str_p,
            )
        )

    # }}}

    # {{{ Intra-group

    # Intra-group scheds would be same due to lbarrier,
    # but since lex tuples are not simplified in intra-group/global
    # cases, there's an extra lex dim:

    sched_stmt_1_intra_group_exp = isl.Map(
        "[n] -> {"
        "[%s=0, i, j, l0] -> [%s] : "
        "%s and %s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["1", "j", "0"],  # lex points
                lid_inames=["l0"],
                ),
            ij_bound_str,
            conc_iname_bound_str,
            )
        )

    sched_stmt_2_intra_group_exp = isl.Map(
        "[n] -> {"
        "[%s=1, i, j, l0] -> [%s] : "
        "%s and %s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["1", "j", "1"],  # lex points
                lid_inames=["l0"],
                ),
            ij_bound_str,
            conc_iname_bound_str,
            )
        )

    sio_intra_group_exp = _isl_map_with_marked_dims(
        "[n] -> {{ "
        "[{0}'=0, i', j', l0'] -> [{0}=1, i, j, l0] : "
        "j' <= j "
        "and {1} and {2} and {3} and {4}"  # iname bounds
        "}}".format(
            STATEMENT_VAR_NAME,
            ij_bound_str,
            ij_bound_str_p,
            conc_iname_bound_str,
            conc_iname_bound_str_p,
            )
        )

    # }}}

    # {{{ Global

    sched_stmt_1_global_exp = isl.Map(
        "[n] -> {"
        "[%s=0, i, j, l0] -> [%s] : "
        "%s and %s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["0"],  # lex points
                lid_inames=["l0"],
                ),
            ij_bound_str,
            conc_iname_bound_str,
            )
        )

    # (same as stmt_1 except for statement id because no global barriers)
    sched_stmt_2_global_exp = isl.Map(
        "[n] -> {"
        "[%s=1, i, j, l0] -> [%s] : "
        "%s and %s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["0"],  # lex points
                lid_inames=["l0"],
                ),
            ij_bound_str,
            conc_iname_bound_str,
            )
        )

    sio_global_exp = _isl_map_with_marked_dims(
        "[n] -> {{ "
        "[{0}'=0, i', j', l0'] -> [{0}=1, i, j, l0] : "
        "False "
        "and {1} and {2} and {3} and {4}"  # iname bounds
        "}}".format(
            STATEMENT_VAR_NAME,
            ij_bound_str,
            ij_bound_str_p,
            conc_iname_bound_str,
            conc_iname_bound_str_p,
            )
        )

    # }}}

    _check_orderings_for_stmt_pair(
        "stmt_1", "stmt_2", pworders,
        sio_intra_thread_exp=sio_intra_thread_exp,
        sched_before_intra_thread_exp=sched_stmt_1_intra_thread_exp,
        sched_after_intra_thread_exp=sched_stmt_2_intra_thread_exp,
        sio_intra_group_exp=sio_intra_group_exp,
        sched_before_intra_group_exp=sched_stmt_1_intra_group_exp,
        sched_after_intra_group_exp=sched_stmt_2_intra_group_exp,
        sio_global_exp=sio_global_exp,
        sched_before_global_exp=sched_stmt_1_global_exp,
        sched_after_global_exp=sched_stmt_2_global_exp,
        )

    # }}}

# }}}


# {{{ test_sios_with_matmul

def test_sios_with_matmul():
    from loopy.schedule.checker import (
        get_pairwise_statement_orderings,
    )
    # For now, this test just ensures all pairwise SIOs can be created
    # for a complex parallel kernel without any errors/exceptions. Later PRs
    # will examine this kernel's SIOs and related dependencies for accuracy.

    bsize = 16
    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],
            name="matmul",
            assumptions="n,m,ell >= 1",
            lang_version=(2018, 2),
            )
    knl = lp.add_and_infer_dtypes(
        knl, dict(a=np.float32, b=np.float32))
    knl = lp.split_iname(
        knl, "i", bsize, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_iname(
        knl, "j", bsize, outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_iname(knl, "k", bsize)
    knl = lp.add_prefetch(
        knl, "a", ["k_inner", "i_inner"], default_tag="l.auto")
    knl = lp.add_prefetch(
        knl, "b", ["j_inner", "k_inner"], default_tag="l.auto")
    knl = lp.prioritize_loops(knl, "k_outer,k_inner")

    # Get a linearization
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    # Get ALL statement id pairs
    all_stmt_ids = _get_runinstruction_ids_from_linearization(lin_items)

    from itertools import product
    stmt_id_pairs = []
    for idx, sid in enumerate(all_stmt_ids):
        stmt_id_pairs.extend(product([sid], all_stmt_ids[idx+1:]))

    # Generate pairwise ordering info for every pair
    get_pairwise_statement_orderings(
        lin_knl, lin_items, stmt_id_pairs)

# }}}


# {{{ Dependency tests

# {{{ Helper functions


def _compare_dependencies(knl, deps_expected, return_unsatisfied=False):

    deps_found = {}
    for stmt in knl.instructions:
        if hasattr(stmt, "dependencies") and stmt.dependencies:
            deps_found[stmt.id] = stmt.dependencies

    assert deps_found.keys() == deps_expected.keys()

    for stmt_id_after, dep_dict_found in deps_found.items():

        dep_dict_expected = deps_expected[stmt_id_after]

        # Ensure deps for stmt_id_after match
        assert dep_dict_found.keys() == dep_dict_expected.keys()

        for stmt_id_before, dep_list_found in dep_dict_found.items():

            # Ensure deps from (stmt_id_before -> stmt_id_after) match
            dep_list_expected = dep_dict_expected[stmt_id_before]
            assert len(dep_list_found) == len(dep_list_expected)
            _align_and_compare_maps(zip(dep_list_found, dep_list_expected))

    if not return_unsatisfied:
        return

    # Get unsatisfied deps
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)
    return lp.find_unsatisfied_dependencies(proc_knl, lin_items)


# }}}


# {{{ Dependency creation and checking (without transformations)

# {{{ test_add_dependency_v2

def test_add_dependency_v2():

    # Make kernel and use OLD deps to control linearization order for now
    i_range_str = "0 <= i < pi"
    i_range_str_p = "0 <= i' < pi"
    assumptions_str = "pi >= 1"
    knl = lp.make_kernel(
        "{[i]: %s}" % (i_range_str),
        """
        a[i] = 3.14  {id=stmt_a}
        b[i] = a[i]  {id=stmt_b, dep=stmt_a}
        c[i] = b[i]  {id=stmt_c, dep=stmt_b}
        """,
        name="example",
        assumptions=assumptions_str,
        lang_version=(2018, 2)
        )
    knl = lp.add_and_infer_dtypes(
            knl, {"a": np.float32, "b": np.float32, "c": np.float32})

    for stmt in knl.instructions:
        assert not stmt.dependencies

    # Add a dependency to stmt_b
    dep_b_on_a = _isl_map_with_marked_dims(
        "[pi] -> {{ [{0}'=0, i'] -> [{0}=1, i] : i > i' "
        "and {1} and {2} and {3} }}".format(
            STATEMENT_VAR_NAME,
            i_range_str,
            i_range_str_p,
            assumptions_str,
            ))

    knl = lp.add_dependency_v2(knl, "stmt_b", "stmt_a", dep_b_on_a)

    _compare_dependencies(
        knl,
        {"stmt_b": {
            "stmt_a": [dep_b_on_a, ]}})

    # Add a second dependency to stmt_b
    dep_b_on_a_2 = _isl_map_with_marked_dims(
        "[pi] -> {{ [{0}'=0, i'] -> [{0}=1, i] : i = i' "
        "and {1} and {2} and {3} }}".format(
            STATEMENT_VAR_NAME,
            i_range_str,
            i_range_str_p,
            assumptions_str,
            ))

    knl = lp.add_dependency_v2(knl, "stmt_b", "stmt_a", dep_b_on_a_2)

    _compare_dependencies(
        knl,
        {"stmt_b": {
            "stmt_a": [dep_b_on_a, dep_b_on_a_2]}})

    # Add dependencies to stmt_c

    dep_c_on_a = _isl_map_with_marked_dims(
        "[pi] -> {{ [{0}'=0, i'] -> [{0}=1, i] : i >= i' "
        "and {1} and {2} and {3} }}".format(
            STATEMENT_VAR_NAME,
            i_range_str,
            i_range_str_p,
            assumptions_str,
            ))
    dep_c_on_b = _isl_map_with_marked_dims(
        "[pi] -> {{ [{0}'=0, i'] -> [{0}=1, i] : i >= i' "
        "and {1} and {2} and {3} }}".format(
            STATEMENT_VAR_NAME,
            i_range_str,
            i_range_str_p,
            assumptions_str,
            ))

    knl = lp.add_dependency_v2(knl, "stmt_c", "stmt_a", dep_c_on_a)
    knl = lp.add_dependency_v2(knl, "stmt_c", "stmt_b", dep_c_on_b)

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {
            "stmt_b": {
                "stmt_a": [dep_b_on_a, dep_b_on_a_2]},
            "stmt_c": {
                "stmt_a": [dep_c_on_a, ], "stmt_b": [dep_c_on_b, ]},
        },
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}


# {{{ test_new_dependencies_finite_diff

def test_new_dependencies_finite_diff():

    # Define kernel
    knl = lp.make_kernel(
        "[nx,nt] -> {[x, t]: 0<=x<nx and 0<=t<nt}",
        "u[t+2,x+1] = 2*u[t+1,x+1] + dt**2/dx**2 "
        "* (u[t+1,x+2] - 2*u[t+1,x+1] + u[t+1,x]) - u[t,x+1]  {id=stmt}")
    knl = lp.add_dtypes(
        knl, {"u": np.float32, "dx": np.float32, "dt": np.float32})

    # Define dependency
    xt_range_str = "0 <= x < nx and 0 <= t < nt"
    xt_range_str_p = "0 <= x' < nx and 0 <= t' < nt"
    dep = _isl_map_with_marked_dims(
        "[nx,nt] -> {{ [{0}'=0, x', t'] -> [{0}=0, x, t] : "
        "((x = x' and t = t'+2) or "
        " (x'-1 <= x <= x'+1 and t = t' + 1)) and "
        "{1} and {2} }}".format(
            STATEMENT_VAR_NAME,
            xt_range_str,
            xt_range_str_p,
            ))
    knl = lp.add_dependency_v2(knl, "stmt", "stmt", dep)

    ref_knl = knl

    # {{{ Check with corrct loop nest order

    # Prioritize loops correctly
    knl = lp.prioritize_loops(knl, "t,x")

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmt": {"stmt": [dep, ]}, },
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}
    # {{{ Check with incorrect loop nest order

    # Now prioritize loops incorrectly
    knl = ref_knl
    knl = lp.prioritize_loops(knl, "x,t")

    # Compare deps and make sure unsatisfied deps are caught
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmt": {"stmt": [dep, ]}, },
        return_unsatisfied=True)

    assert len(unsatisfied_deps) == 1

    # }}}
    # {{{ Check with parallel x and no barrier

    # Parallelize the x loop
    knl = ref_knl
    knl = lp.prioritize_loops(knl, "t,x")
    knl = lp.tag_inames(knl, "x:l.0")

    # Make sure unsatisfied deps are caught
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    # Without a barrier, deps not satisfied
    # Make sure there is no barrier, and that unsatisfied deps are caught
    from loopy.schedule import Barrier
    print(lp.generate_code_v2(lin_knl).device_code())
    for lin_item in lin_items:
        assert not isinstance(lin_item, Barrier)

    unsatisfied_deps = lp.find_unsatisfied_dependencies(
        proc_knl, lin_items)

    assert len(unsatisfied_deps) == 1

    # }}}
    # {{{ Check with parallel x and included barrier

    # Insert a barrier to satisfy deps
    knl = lp.make_kernel(
        "[nx,nt] -> {[x, t]: 0<=x<nx and 0<=t<nt}",
        """
        for x,t
            ...lbarrier
            u[t+2,x+1] = 2*u[t+1,x+1] + dt**2/dx**2 \
                *(u[t+1,x+2] - 2*u[t+1,x+1] + u[t+1,x]) - u[t,x+1]  {id=stmt}
        end
        """)
    knl = lp.add_dtypes(
        knl, {"u": np.float32, "dx": np.float32, "dt": np.float32})

    knl = lp.add_dependency_v2(knl, "stmt", "stmt", dep)

    knl = lp.prioritize_loops(knl, "t,x")
    knl = lp.tag_inames(knl, "x:l.0")

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmt": {"stmt": [dep, ]}, },
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}

    # Transformations to test after dep handling during transformation:
    # knl = lp.split_iname(knl, "x", 14)
    # knl = lp.assume(knl, "nx % 14 = 0 and nt >= 1 and nx >= 1")
    # knl = lp.tag_inames(knl, "x_outer:g.0, x_inner:l.0")

# }}}

# }}}


# {{{ Dependency handling during transformations

# {{{ test_fix_parameters_with_dependencies

def test_fix_parameters_with_dependencies():
    knl = lp.make_kernel(
        "{[i,j]: 0 <= i < n and 0 <= j < m}",
        """
        <>temp0 = 0.1*i+j {id=stmt0}
        <>tsq = temp0**2+i+j  {id=stmt1,dep=stmt0}
        a[i,j] = 23*tsq + 25*tsq+j  {id=stmt2,dep=stmt1}
        """)

    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})

    dep_orig = _isl_map_with_marked_dims(
        "[n,m] -> {{ [{0}'=0, i', j']->[{0}=1, i, j] : "
        "0 <= i,i' < n and 0 <= j,j' < m "
        "and i' = i and j' = j"
        "}}".format(STATEMENT_VAR_NAME))

    from copy import deepcopy
    knl = lp.add_dependency_v2(knl, "stmt1", "stmt0", deepcopy(dep_orig))
    knl = lp.add_dependency_v2(knl, "stmt2", "stmt1", deepcopy(dep_orig))

    fix_val = 64
    knl = lp.fix_parameters(knl, m=fix_val)

    dep_exp = _isl_map_with_marked_dims(
        "[n] -> {{ [{0}'=0, i', j']->[{0}=1, i, j] : "
        "0 <= i,i' < n and 0 <= j,j' < {1} "
        "and i' = i and j' = j"
        "}}".format(STATEMENT_VAR_NAME, fix_val))

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {
            "stmt1": {"stmt0": [dep_exp, ]},
            "stmt2": {"stmt1": [dep_exp, ]},
        },
        return_unsatisfied=True)

    assert not unsatisfied_deps

# }}}


# {{{ test_assignment_to_subst_with_dependencies

def test_assignment_to_subst_with_dependencies():
    knl = lp.make_kernel(
        "{[i]: 0 <= i < n}",
        """
        <>temp0 = 0.1*i {id=stmt0}
        <>tsq = temp0**2  {id=stmt1,dep=stmt0}
        a[i] = 23*tsq + 25*tsq  {id=stmt2,dep=stmt1}
        <>temp3 = 3*tsq  {id=stmt3,dep=stmt1}
        <>temp4 = 5.5*i {id=stmt4,dep=stmt1}
        """)

    # TODO test with multiple subst definition sites
    # TODO what if stmt2 depends on <>tsq = b[i-1]**2 and then we do
    #     assignment to subst? remove i'=i from dep?
    # TODO what if, e.g., stmt3 doesn't have iname i in it?
    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})

    dep_eq = _isl_map_with_marked_dims(
        "[n] -> {{ [{0}'=0, i']->[{0}=1, i] : "
        "0 <= i,i' < n and i' = i"
        "}}".format(STATEMENT_VAR_NAME))
    dep_le = _isl_map_with_marked_dims(
        "[n] -> {{ [{0}'=0, i']->[{0}=1, i] : "
        "0 <= i,i' < n and i' <= i"
        "}}".format(STATEMENT_VAR_NAME))

    from copy import deepcopy
    knl = lp.add_dependency_v2(knl, "stmt1", "stmt0", deepcopy(dep_le))
    knl = lp.add_dependency_v2(knl, "stmt2", "stmt1", deepcopy(dep_eq))
    knl = lp.add_dependency_v2(knl, "stmt3", "stmt1", deepcopy(dep_eq))
    knl = lp.add_dependency_v2(knl, "stmt4", "stmt1", deepcopy(dep_eq))

    knl = lp.assignment_to_subst(knl, "tsq")

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {
            "stmt2": {"stmt0": [dep_le, ]},
            "stmt3": {"stmt0": [dep_le, ]},
        },
        return_unsatisfied=True)
    # (stmt4 dep was removed because dependee was removed, but dependee's
    # deps were not added to stmt4 because the substitution was not made
    # in stmt4) TODO this behavior will change when we propagate deps properly

    assert not unsatisfied_deps

    # Test using 'within' --------------------------------------------------

    knl = lp.make_kernel(
        "{[i]: 0 <= i < n}",
        """
        <>temp0 = 0.1*i {id=stmt0}
        <>tsq = temp0**2  {id=stmt1,dep=stmt0}
        a[i] = 23*tsq + 25*tsq  {id=stmt2,dep=stmt1}
        <>temp3 = 3*tsq  {id=stmt3,dep=stmt1}
        <>temp4 = 5.5*i {id=stmt4,dep=stmt1}
        <>temp5 = 5.6*tsq*i {id=stmt5,dep=stmt1}
        """)

    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})

    knl = lp.add_dependency_v2(knl, "stmt1", "stmt0", deepcopy(dep_le))
    knl = lp.add_dependency_v2(knl, "stmt2", "stmt1", deepcopy(dep_eq))
    knl = lp.add_dependency_v2(knl, "stmt3", "stmt1", deepcopy(dep_eq))
    knl = lp.add_dependency_v2(knl, "stmt4", "stmt1", deepcopy(dep_eq))
    knl = lp.add_dependency_v2(knl, "stmt5", "stmt1", deepcopy(dep_eq))

    knl = lp.assignment_to_subst(knl, "tsq", within="id:stmt2 or id:stmt3")

    # Replacement will not be made in stmt5, so stmt1 will not be removed,
    # which means no deps will be removed, and the statements where the replacement
    # *was* made (stmt2 and stmt3) will still receive the deps from stmt1
    # TODO this behavior may change when we propagate deps properly

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {
            "stmt1": {"stmt0": [dep_le, ]},
            "stmt2": {
                "stmt0": [dep_le, ], "stmt1": [dep_eq, ]},
            "stmt3": {
                "stmt0": [dep_le, ], "stmt1": [dep_eq, ]},
            "stmt4": {"stmt1": [dep_eq, ]},
            "stmt5": {"stmt1": [dep_eq, ]},
        },
        return_unsatisfied=True)

    assert not unsatisfied_deps

# }}}


# {{{ test_duplicate_inames_with_dependencies

def test_duplicate_inames_with_dependencies():

    knl = lp.make_kernel(
        "{[i,j]: 0 <= i,j < n}",
        """
        b[i,j] = a[i,j]  {id=stmtb}
        c[i,j] = a[i,j]  {id=stmtc,dep=stmtb}
        """)
    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})

    dep_eq = _isl_map_with_marked_dims(
        "[n] -> {{ [{0}'=0, i', j']->[{0}=1, i, j] : "
        "0 <= i,i',j,j' < n and i' = i and j' = j"
        "}}".format(STATEMENT_VAR_NAME))

    # Create dep stmtb->stmtc
    knl = lp.add_dependency_v2(knl, "stmtc", "stmtb", dep_eq)

    ref_knl = knl

    # {{{ Duplicate j within stmtc

    knl = lp.duplicate_inames(knl, ["j"], within="id:stmtc", new_inames=["j_new"])

    dep_exp = _isl_map_with_marked_dims(
        "[n] -> {{ [{0}'=0, i', j']->[{0}=1, i, j_new] : "
        "0 <= i,i',j_new,j' < n and i' = i and j' = j_new"
        "}}".format(STATEMENT_VAR_NAME))

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmtc": {"stmtb": [dep_exp, ]}},
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}

    # {{{ Duplicate j within stmtb

    knl = ref_knl
    knl = lp.duplicate_inames(knl, ["j"], within="id:stmtb", new_inames=["j_new"])

    dep_exp = _isl_map_with_marked_dims(
        "[n] -> {{ [{0}'=0, i', j_new']->[{0}=1, i, j] : "
        "0 <= i,i',j,j_new' < n and i' = i and j_new' = j"
        "}}".format(STATEMENT_VAR_NAME))

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmtc": {"stmtb": [dep_exp, ]}},
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}

    # {{{ Duplicate j within stmtb and stmtc

    knl = ref_knl
    knl = lp.duplicate_inames(
        knl, ["j"], within="id:stmtb or id:stmtc", new_inames=["j_new"])

    dep_exp = _isl_map_with_marked_dims(
        "[n] -> {{ [{0}'=0, i', j_new']->[{0}=1, i, j_new] : "
        "0 <= i,i',j_new,j_new' < n and i' = i and j_new' = j_new"
        "}}".format(STATEMENT_VAR_NAME))

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmtc": {"stmtb": [dep_exp, ]}},
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}

# }}}


# {{{ test_split_iname_with_dependencies

def test_split_iname_with_dependencies():
    knl = lp.make_kernel(
        "{[i]: 0<=i<p}",
        """
        a[i] = 0.1  {id=stmt0}
        b[i] = a[i]  {id=stmt1,dep=stmt0}
        """,
        name="example",
        assumptions="p >= 1",
        lang_version=(2018, 2)
        )

    from copy import deepcopy
    ref_knl = deepcopy(knl)  # without deepcopy, deps get applied to ref_knl

    # {{{ Split iname and make sure dep is correct

    dep_inout_space_str = "[{0}'=0, i'] -> [{0}=1, i]".format(STATEMENT_VAR_NAME)
    dep_satisfied = _isl_map_with_marked_dims(
        "[p] -> { %s : 0 <= i < p and i' = i }"
        % (dep_inout_space_str))

    knl = lp.add_dependency_v2(knl, "stmt1", "stmt0", dep_satisfied)
    knl = lp.split_iname(knl, "i", 32)

    dep_exp = _isl_map_with_marked_dims(
        "[p] -> {{ [{0}'=0, i_outer', i_inner'] -> [{0}=1, i_outer, i_inner] : "
        "0 <= i_inner, i_inner' < 32"  # new bounds
        " and 0 <= 32*i_outer + i_inner < p"  # transformed bounds (0 <= i < p)
        " and 0 <= 32*i_outer' + i_inner' < p"  # transformed bounds (0 <= i' < p)
        " and i_inner + 32*i_outer = 32*i_outer' + i_inner'"  # i = i'
        "}}".format(STATEMENT_VAR_NAME))

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmt1": {"stmt0": [dep_exp, ]}},
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}

    # {{{ Split iname within stmt1 and make sure dep is correct

    knl = deepcopy(ref_knl)

    knl = lp.add_dependency_v2(knl, "stmt1", "stmt0", dep_satisfied)
    knl = lp.split_iname(knl, "i", 32, within="id:stmt1")

    dep_exp = _isl_map_with_marked_dims(
        "[p] -> {{ [{0}'=0, i'] -> [{0}=1, i_outer, i_inner] : "
        "0 <= i_inner < 32"  # new bounds
        " and 0 <= 32*i_outer + i_inner < p"  # transformed bounds (0 <= i < p)
        " and 0 <= i' < p"  # original bounds
        " and i_inner + 32*i_outer = i'"  # transform {i = i'}
        "}}".format(STATEMENT_VAR_NAME))

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmt1": {"stmt0": [dep_exp, ]}},
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}

    # {{{ Split iname within stmt0 and make sure dep is correct

    knl = deepcopy(ref_knl)

    knl = lp.add_dependency_v2(knl, "stmt1", "stmt0", dep_satisfied)
    knl = lp.split_iname(knl, "i", 32, within="id:stmt0")

    dep_exp = _isl_map_with_marked_dims(
        "[p] -> {{ [{0}'=0, i_outer', i_inner'] -> [{0}=1, i] : "
        "0 <= i_inner' < 32"  # new bounds
        " and 0 <= i < p"  # original bounds
        " and 0 <= 32*i_outer' + i_inner' < p"  # transformed bounds (0 <= i' < p)
        " and i = 32*i_outer' + i_inner'"  # transform {i = i'}
        "}}".format(STATEMENT_VAR_NAME))

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmt1": {"stmt0": [dep_exp, ]}},
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}

    # {{{ Check dep that should not be satisfied

    knl = deepcopy(ref_knl)

    dep_unsatisfied = _isl_map_with_marked_dims(
        "[p] -> { %s : 0 <= i < p and i' = i + 1 }"
        % (dep_inout_space_str))

    knl = lp.add_dependency_v2(knl, "stmt1", "stmt0", dep_unsatisfied)
    knl = lp.split_iname(knl, "i", 32)

    dep_exp = _isl_map_with_marked_dims(
        "[p] -> {{ [{0}'=0, i_outer', i_inner'] -> [{0}=1, i_outer, i_inner] : "
        "0 <= i_inner, i_inner' < 32"  # new bounds
        " and 0 <= 32*i_outer + i_inner < p"  # transformed bounds (0 <= i < p)
        " and 0 <= 32*i_outer' + i_inner' - 1 < p"  # trans. bounds (0 <= i'-1 < p)
        " and i_inner + 32*i_outer + 1 = 32*i_outer' + i_inner'"  # i' = i + 1
        "}}".format(STATEMENT_VAR_NAME))

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmt1": {"stmt0": [dep_exp, ]}},
        return_unsatisfied=True)

    assert len(unsatisfied_deps) == 1

    # }}}

    # {{{ Deps that should be satisfied after gratuitous splitting

    knl = lp.make_kernel(
        "{[i,j,k,m]: 0<=i,j,k,m<p}",
        """
        a[i,k] = 0.1  {id=stmt0}
        b[i,k] = a[i,k]  {id=stmt1,dep=stmt0}
        c[i,k,j,m] = 0.1  {id=stmt2}
        d[i,k,j,m] = c[i,k,j,m]  {id=stmt3,dep=stmt2}
        """,
        name="example",
        assumptions="p >= 1",
        lang_version=(2018, 2)
        )

    dep_ik_space_str = "[{0}'=0, i', k'] -> [{0}=1, i, k]".format(
        STATEMENT_VAR_NAME)
    dep_ijkm_space_str = "[{0}'=0, i', j', k', m'] -> [{0}=1, i, j, k, m]".format(
        STATEMENT_VAR_NAME)
    #iname_bounds_str = "0 <= i,j,k,m,i',j',k',m' < p"
    ik_bounds_str = "0 <= i,k,i',k' < p"
    ijkm_bounds_str = ik_bounds_str + " and 0 <= j,m,j',m' < p"
    dep_stmt1_on_stmt0_eq = _isl_map_with_marked_dims(
        "[p] -> { %s : %s and i' = i and k' = k}"
        % (dep_ik_space_str, ik_bounds_str))
    dep_stmt1_on_stmt0_lt = _isl_map_with_marked_dims(
        "[p] -> { %s : %s and i' < i and k' < k}"
        % (dep_ik_space_str, ik_bounds_str))
    dep_stmt3_on_stmt2_eq = _isl_map_with_marked_dims(
        "[p] -> { %s : %s and i' = i and k' = k and j' = j and m' = m}"
        % (dep_ijkm_space_str, ijkm_bounds_str))

    knl = lp.add_dependency_v2(knl, "stmt1", "stmt0", dep_stmt1_on_stmt0_eq)
    knl = lp.add_dependency_v2(knl, "stmt1", "stmt0", dep_stmt1_on_stmt0_lt)
    knl = lp.add_dependency_v2(knl, "stmt3", "stmt2", dep_stmt3_on_stmt2_eq)

    # Gratuitous splitting
    knl = lp.split_iname(knl, "i", 64)
    knl = lp.split_iname(knl, "j", 64)
    knl = lp.split_iname(knl, "k", 64)
    knl = lp.split_iname(knl, "m", 64)
    knl = lp.split_iname(knl, "i_inner", 8)
    knl = lp.split_iname(knl, "j_inner", 8)
    knl = lp.split_iname(knl, "k_inner", 8)
    knl = lp.split_iname(knl, "m_inner", 8)
    knl = lp.split_iname(knl, "i_outer", 4)
    knl = lp.split_iname(knl, "j_outer", 4)
    knl = lp.split_iname(knl, "k_outer", 4)
    knl = lp.split_iname(knl, "m_outer", 4)

    # Get a linearization
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    unsatisfied_deps = lp.find_unsatisfied_dependencies(
        proc_knl, lin_items)

    assert not unsatisfied_deps

    # }}}

# }}}


# {{{ test map domain with dependencies

# {{{ test_map_domain_with_only_partial_dep_pair_affected

def test_map_domain_with_only_partial_dep_pair_affected():

    # Split an iname using map_domain, and have (misaligned) deps
    # where only the dependee uses the split iname

    # {{{ Make kernel

    knl = lp.make_kernel(
        [
            "[nx,nt] -> {[x, t]: 0 <= x < nx and 0 <= t < nt}",
            "[ni] -> {[i]: 0 <= i < ni}",
        ],
        """
        a[x,t] = b[x,t]  {id=stmta}
        c[x,t] = d[x,t]  {id=stmtc,dep=stmta}
        e[i] = f[i]  {id=stmte,dep=stmtc}
        """,
        name="wave_equation",
        lang_version=(2018, 2),
        )
    knl = lp.add_and_infer_dtypes(knl, {"b,d,f": np.float32})

    # }}}

    # {{{ Add dependencies

    dep_c_on_a = _isl_map_with_marked_dims(
        "[nx, nt] -> {{"
        "[{0}' = 0, x', t'] -> [{0} = 1, x, t] : "
        "0 <= x,x' < nx and 0 <= t,t' < nt and "
        "t' <= t and x' <= x"
        "}}".format(STATEMENT_VAR_NAME))

    knl = lp.add_dependency_v2(
        knl, "stmtc", "stmta", dep_c_on_a)

    # Intentionally make order of x and t different from transform_map below
    # to test alignment steps in map_domain
    dep_e_on_c = _isl_map_with_marked_dims(
        "[nx, nt, ni] -> {{"
        "[{0}' = 0, t', x'] -> [{0} = 1, i] : "
        "0 <= x' < nx and 0 <= t' < nt and 0 <= i < ni"
        "}}".format(STATEMENT_VAR_NAME))

    knl = lp.add_dependency_v2(
        knl, "stmte", "stmtc", dep_e_on_c)

    # }}}

    # {{{ Apply domain change mapping

    # Create map_domain mapping:
    import islpy as isl
    transform_map = isl.BasicMap(
        "[nx,nt] -> {[x, t] -> [x_, t_outer, t_inner]: "
        "x = x_ and "
        "0 <= t_inner < 32 and "
        "32*t_outer + t_inner = t and "
        "0 <= 32*t_outer + t_inner < nt}")

    # Call map_domain to transform kernel
    knl = lp.map_domain(knl, transform_map, rename_after={"x_": "x"})

    # Prioritize loops (prio should eventually be updated in map_domain?)
    knl = lp.prioritize_loops(knl, "x, t_outer, t_inner")

    # }}}

    # {{{ Create expected dependencies

    dep_c_on_a_exp = _isl_map_with_marked_dims(
        "[nx, nt] -> {{"
        "[{0}' = 0, x', t_outer', t_inner'] -> [{0} = 1, x, t_outer, t_inner] : "
        "0 <= x,x' < nx and "  # old bounds
        "0 <= t_inner,t_inner' < 32 and "  # new bounds
        "0 <= 32*t_outer + t_inner < nt and "  # new bounds
        "0 <= 32*t_outer' + t_inner' < nt and "  # new bounds
        "32*t_outer' + t_inner' <= 32*t_outer + t_inner and "  # new constraint t'<=t
        "x' <= x"  # old constraint
        "}}".format(STATEMENT_VAR_NAME))

    dep_e_on_c_exp = _isl_map_with_marked_dims(
        "[nx, nt, ni] -> {{"
        "[{0}' = 0, x', t_outer', t_inner'] -> [{0} = 1, i] : "
        "0 <= x' < nx and 0 <= i < ni and "  # old bounds
        "0 <= t_inner' < 32 and "  # new bounds
        "0 <= 32*t_outer' + t_inner' < nt"  # new bounds
        "}}".format(STATEMENT_VAR_NAME))

    # }}}

    # {{{ Make sure deps are correct and satisfied

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {
            "stmtc": {
                "stmta": [dep_c_on_a_exp, ]},
            "stmte": {
                "stmtc": [dep_e_on_c_exp, ]},
        },
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}

# }}}


# {{{ test_map_domain_with_stencil_dependencies

def test_map_domain_with_stencil_dependencies():

    # {{{ Make kernel

    knl = lp.make_kernel(
        "[nx,nt] -> {[ix, it]: 1<=ix<nx-1 and 0<=it<nt}",
        """
        u[ix, it+2] = (
            2*u[ix, it+1]
            + dt**2/dx**2 * (u[ix+1, it+1] - 2*u[ix, it+1] + u[ix-1, it+1])
            - u[ix, it])  {id=stmt}
        """,
        name="wave_equation",
        #assumptions="nx,nt >= 3",  # works without these (?)
        lang_version=(2018, 2),
        )
    knl = lp.add_and_infer_dtypes(knl, {"u,dt,dx": np.float32})
    stmt_before = stmt_after = "stmt"

    # }}}

    # {{{ Add dependency

    dep_map = _isl_map_with_marked_dims(
        "[nx, nt] -> {{"
        "[{0}' = 0, ix', it'] -> [{0} = 0, ix, it = 1 + it'] : "
        "0 < ix' <= -2 + nx and 0 <= it' <= -2 + nt and ix >= -1 + ix' and "
        "0 < ix <= 1 + ix' and ix <= -2 + nx; "
        "[statement' = 0, ix', it'] -> [statement = 0, ix = ix', it = 2 + it'] : "
        "0 < ix' <= -2 + nx and 0 <= it' <= -3 + nt"
        "}}".format(STATEMENT_VAR_NAME))

    knl = lp.add_dependency_v2(
        knl, stmt_after, stmt_before, dep_map)

    # }}}

    # {{{ Check deps *without* map_domain transformation

    ref_knl = knl

    # Prioritize loops
    knl = lp.prioritize_loops(knl, ("it", "ix"))  # valid
    #knl = lp.prioritize_loops(knl, ("ix", "it"))  # invalid

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmt": {"stmt": [dep_map, ]}},
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}

    # {{{ Check dependency after domain change mapping

    knl = ref_knl  # loop priority goes away, deps stay

    # Create map_domain mapping:
    transform_map = isl.BasicMap(
        "[nx,nt] -> {[ix, it] -> [tx, tt, tparity, itt, itx]: "
        "16*(tx - tt) + itx - itt = ix - it and "
        "16*(tx + tt + tparity) + itt + itx = ix + it and "
        "0<=tparity<2 and 0 <= itx - itt < 16 and 0 <= itt+itx < 16}")

    # Call map_domain to transform kernel
    knl = lp.map_domain(knl, transform_map)

    # Prioritize loops (prio should eventually be updated in map_domain?)
    knl = lp.prioritize_loops(knl, "tt,tparity,tx,itt,itx")

    # {{{ Create expected dependency

    # Prep transform map to be applied to dependency
    from loopy.schedule.checker.utils import (
        insert_and_name_isl_dims,
        add_eq_isl_constraint_from_names,
        append_mark_to_isl_map_var_names,
    )
    dt = isl.dim_type
    # Insert 'statement' dim into transform map
    transform_map = insert_and_name_isl_dims(
            transform_map, dt.in_, [STATEMENT_VAR_NAME+BEFORE_MARK], 0)
    transform_map = insert_and_name_isl_dims(
            transform_map, dt.out, [STATEMENT_VAR_NAME], 0)
    # Add stmt = stmt' constraint
    transform_map = add_eq_isl_constraint_from_names(
        transform_map, STATEMENT_VAR_NAME, STATEMENT_VAR_NAME+BEFORE_MARK)

    # Apply transform map to dependency
    mapped_dep_map = dep_map.apply_range(transform_map).apply_domain(transform_map)
    mapped_dep_map = append_mark_to_isl_map_var_names(
        mapped_dep_map, dt.in_, BEFORE_MARK)

    # }}}

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmt": {"stmt": [mapped_dep_map, ]}},
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}

# }}}

# }}}

# }}}


# {{{ Dependency handling during linearization

# {{{ test_filtering_deps_by_same

def test_filtering_deps_by_same():

    # Make a kernel (just need something that can carry deps)
    knl = lp.make_kernel(
        "{[i,j,k,m] : 0 <= i,j,k,m < n}",
        """
        a[i,j,k,m] = 5 {id=s5}
        a[i,j,k,m] = 4 {id=s4}
        a[i,j,k,m] = 3 {id=s3}
        a[i,j,k,m] = 2 {id=s2}
        a[i,j,k,m] = 1 {id=s1}
        """)
    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})
    knl = lp.tag_inames(knl, "m:l.0")

    # Make some deps

    def _dep_with_condition(stmt_before, stmt_after, cond):
        sid_after = 0 if stmt_before == stmt_after else 1
        return _isl_map_with_marked_dims(
            "[n] -> {{"
            "[{0}'=0, i', j', k', m'] -> [{0}={1}, i, j, k, m] : "
            "0 <= i,j,k,m,i',j',k',m' < n and {2}"
            "}}".format(
                STATEMENT_VAR_NAME, sid_after, cond))

    dep_s2_on_s1_1 = _dep_with_condition(2, 1, "i'< i and j'<=j and k'=k and m'<m")
    dep_s2_on_s1_2 = _dep_with_condition(2, 1, "i'<=i and j'<=j and k'=k and m'<m")

    dep_s2_on_s2_1 = _dep_with_condition(2, 2, "i'< i and j'<=j and k'=k and m'<m")
    dep_s2_on_s2_2 = _dep_with_condition(2, 2, "i'<=i and j'<=j and k'=k and m'<m")

    dep_s3_on_s2_1 = _dep_with_condition(3, 2, "i'< i and j'< j and k'=k and m'<m")
    dep_s3_on_s2_2 = _dep_with_condition(3, 2, "i' =i and j'= j and k'<k and m'<m")

    dep_s4_on_s3_1 = _dep_with_condition(4, 3, "i'<=i and j'<=j and k'=k")
    dep_s4_on_s3_2 = _dep_with_condition(4, 3, "i'<=i")

    dep_s5_on_s4_1 = _dep_with_condition(5, 4, "i'< i")

    dep_s5_on_s2_1 = _dep_with_condition(5, 2, "i'= i")

    knl = lp.add_dependency_v2(knl, "s2", "s1", dep_s2_on_s1_1)
    knl = lp.add_dependency_v2(knl, "s2", "s1", dep_s2_on_s1_2)

    knl = lp.add_dependency_v2(knl, "s2", "s2", dep_s2_on_s2_1)
    knl = lp.add_dependency_v2(knl, "s2", "s2", dep_s2_on_s2_2)

    knl = lp.add_dependency_v2(knl, "s3", "s2", dep_s3_on_s2_1)
    knl = lp.add_dependency_v2(knl, "s3", "s2", dep_s3_on_s2_2)

    knl = lp.add_dependency_v2(knl, "s4", "s3", dep_s4_on_s3_1)
    knl = lp.add_dependency_v2(knl, "s4", "s3", dep_s4_on_s3_2)

    knl = lp.add_dependency_v2(knl, "s5", "s4", dep_s5_on_s4_1)

    knl = lp.add_dependency_v2(knl, "s5", "s2", dep_s5_on_s2_1)

    # Filter deps by intersection with SAME

    from loopy.schedule.checker.dependency import (
        filter_deps_by_intersection_with_SAME,
    )
    filtered_depends_on_dict = filter_deps_by_intersection_with_SAME(knl)

    # Make sure filtered edges are correct

    # (m is concurrent so shouldn't matter)
    depends_on_dict_expected = {
        "s2": set(["s1", "s2"]),
        "s4": set(["s3"]),
        "s5": set(["s2"]),
        }

    assert filtered_depends_on_dict == depends_on_dict_expected

# }}}


# {{{ test_linearization_using_simplified_dep_graph

def test_linearization_using_simplified_dep_graph():
    # Test use of simplified dep graph inside find_loop_insn_dep_map(),
    # which is called during linearization.
    # The deps created below should yield a simplified dep graph that causes the
    # linearization process to order assignments below in numerical order

    # Make a kernel
    knl = lp.make_kernel(
        "{[i,j,k,m] : 0 <= i,j,k,m < n}",
        """
        for i,j,k,m
            <>t5 = 5 {id=s5}
            <>t3 = 3 {id=s3}
            <>t4 = 4 {id=s4}
            <>t1 = 1 {id=s1}
            <>t2 = 2 {id=s2}
        end
        """)
    knl = lp.tag_inames(knl, "m:l.0")

    stmt_ids_ordered_desired = ["s1", "s2", "s3", "s4", "s5"]

    # {{{ Add some deps

    def _dep_with_condition(stmt_before, stmt_after, cond):
        sid_after = 0 if stmt_before == stmt_after else 1
        return _isl_map_with_marked_dims(
            "[n] -> {{"
            "[{0}'=0, i', j', k', m'] -> [{0}={1}, i, j, k, m] : "
            "0 <= i,j,k,m,i',j',k',m' < n and {2}"
            "}}".format(
                STATEMENT_VAR_NAME, sid_after, cond))

    # Should NOT create an edge:
    dep_s2_on_s1_1 = _dep_with_condition(2, 1, "i'< i and j'<=j and k' =k and m'=m")
    # Should create an edge:
    dep_s2_on_s1_2 = _dep_with_condition(2, 1, "i'<=i and j'<=j and k' =k and m'=m")
    # Should NOT create an edge:
    dep_s2_on_s2_1 = _dep_with_condition(2, 2, "i'< i and j'<=j and k' =k and m'=m")
    # Should NOT create an edge:
    dep_s2_on_s2_2 = _dep_with_condition(2, 2, "i'<=i and j'<=j and k'< k and m'=m")
    # Should create an edge:
    dep_s3_on_s2_1 = _dep_with_condition(3, 2, "i'<=i and j'<=j and k' =k and m'=m")
    # Should create an edge:
    dep_s4_on_s3_1 = _dep_with_condition(4, 3, "i'<=i and j'<=j and k' =k and m'=m")
    # Should create an edge:
    dep_s5_on_s4_1 = _dep_with_condition(5, 4, "i' =i and j' =j and k' =k and m'=m")

    knl = lp.add_dependency_v2(knl, "s2", "s1", dep_s2_on_s1_1)
    knl = lp.add_dependency_v2(knl, "s2", "s1", dep_s2_on_s1_2)
    knl = lp.add_dependency_v2(knl, "s2", "s2", dep_s2_on_s2_1)
    knl = lp.add_dependency_v2(knl, "s2", "s2", dep_s2_on_s2_2)
    knl = lp.add_dependency_v2(knl, "s3", "s2", dep_s3_on_s2_1)
    knl = lp.add_dependency_v2(knl, "s4", "s3", dep_s4_on_s3_1)
    knl = lp.add_dependency_v2(knl, "s5", "s4", dep_s5_on_s4_1)

    # }}}

    # {{{ Test filteringn of deps by intersection with SAME

    from loopy.schedule.checker.dependency import (
        filter_deps_by_intersection_with_SAME,
    )
    filtered_depends_on_dict = filter_deps_by_intersection_with_SAME(knl)

    # Make sure filtered edges are correct

    # (m is concurrent so shouldn't matter)
    depends_on_dict_expected = {
        "s2": set(["s1"]),
        "s3": set(["s2"]),
        "s4": set(["s3"]),
        "s5": set(["s4"]),
        }

    assert filtered_depends_on_dict == depends_on_dict_expected

    # }}}

    # {{{ Get a linearization WITHOUT using the simplified dep graph

    knl = lp.set_options(knl, use_dependencies_v2=False)
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    # Check stmt order (should be wrong)
    stmt_ids_ordered = _get_runinstruction_ids_from_linearization(lin_items)
    assert stmt_ids_ordered != stmt_ids_ordered_desired

    # Check dep satisfaction (should not all be satisfied)
    unsatisfied_deps = lp.find_unsatisfied_dependencies(proc_knl, lin_items)
    assert unsatisfied_deps

    # }}}

    # {{{ Get a linearization using the simplified dep graph

    knl = lp.set_options(knl, use_dependencies_v2=True)
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    # Check stmt order
    stmt_ids_ordered = _get_runinstruction_ids_from_linearization(lin_items)
    assert stmt_ids_ordered == stmt_ids_ordered_desired

    # Check dep satisfaction
    unsatisfied_deps = lp.find_unsatisfied_dependencies(proc_knl, lin_items)
    assert not unsatisfied_deps

    # }}}

# }}}

# }}}

# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

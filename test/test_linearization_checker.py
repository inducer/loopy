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
    make_dep_map,
    prettier_map_string,
)
from loopy.schedule.checker import (
    get_pairwise_statement_orderings,
)
dim_type = isl.dim_type

logger = logging.getLogger(__name__)


# {{{ Helper functions for map creation/handling

def _align_and_compare_maps(maps):

    for map1, map2 in maps:
        # Align maps and compare
        map1_aligned = ensure_dim_names_match_and_align(map1, map2)
        if map1_aligned != map2:
            print("Maps not equal:")
            print(prettier_map_string(map1_aligned))
            print(prettier_map_string(map2))
        assert map1_aligned == map2


def _lex_point_string(dim_vals, lid_inames=(), gid_inames=()):
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
    if BEFORE_MARK == "'":
        # ISL will ignore the apostrophe; manually name the in_ vars
        return append_mark_to_isl_map_var_names(
            isl.Map(s.replace(placeholder_mark, BEFORE_MARK)),
            dim_type.in_,
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


def _process_and_linearize(prog, knl_name="loopy_kernel"):
    # Return linearization items along with the preprocessed kernel and
    # linearized kernel
    proc_prog = preprocess_kernel(prog)
    lin_prog = get_one_linearized_kernel(
        proc_prog[knl_name], proc_prog.callables_table)
    return lin_prog.linearization, proc_prog[knl_name], lin_prog

# }}}


# {{{ Helper functions for dependency tests


def _compare_dependencies(
        prog, deps_expected, return_unsatisfied=False, knl_name="loopy_kernel"):

    deps_found = {}
    for stmt in prog[knl_name].instructions:
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
            print("comparing deps %s->%s" % (stmt_id_before, stmt_id_after))
            assert len(dep_list_found) == len(dep_list_expected)
            _align_and_compare_maps(zip(dep_list_found, dep_list_expected))

    if not return_unsatisfied:
        return

    # Get unsatisfied deps
    lin_items, proc_prog, lin_prog = _process_and_linearize(prog, knl_name)
    unsatisfied_deps = lp.find_unsatisfied_dependencies(
        proc_prog, lin_items, stop_on_first_violation=False)

    # Make sure dep checking also works with just linearized kernel
    unsatisfied_deps_2 = lp.find_unsatisfied_dependencies(
        lin_prog, stop_on_first_violation=False)
    assert len(unsatisfied_deps) == len(unsatisfied_deps_2)

    return unsatisfied_deps

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
        perform_closure_checks=True,
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
        perform_closure_checks=True,
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
        perform_closure_checks=True,
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
        perform_closure_checks=True,
        )

    # Create string for representing parallel iname condition in sio
    conc_inames, _ = partition_inames_by_concurrency(knl["loopy_kernel"])
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


# {{{ test_statement_instance_ordering_of_barriers()

def test_statement_instance_ordering_of_barriers():
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
                ... gbarrier {id=gbar}
                for j
                    for jj
                        <>temp = b[i,ii,j,jj]  {id=stmt_a,dep=gbar}
                        ... lbarrier {id=lbar0,dep=stmt_a}
                        a[i,ii,j,jj] = temp + 1  {id=stmt_b,dep=lbar0}
                        ... lbarrier {id=lbar1,dep=stmt_b}
                    end
                end
            end
        end
        <>temp2 = 0.5  {id=stmt_c,dep=lbar1}
        """,
        assumptions="pi,pj >= 1",
        lang_version=(2018, 2)
        )
    knl = lp.add_and_infer_dtypes(knl, {"a,b": np.float32})
    knl = lp.tag_inames(knl, {"j": "l.0", "i": "g.0"})
    knl = lp.prioritize_loops(knl, "ii,jj")

    # Get a linearization
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    # Get pairwise schedules
    stmt_id_pairs = [
        ("stmt_a", "stmt_b"),
        ("gbar", "stmt_a"),
        ("stmt_b", "lbar1"),
        ("lbar1", "stmt_c"),
        ]
    pworders = get_pairwise_statement_orderings(
        lin_knl,
        lin_items,
        stmt_id_pairs,
        perform_closure_checks=True,
        )

    # Create string for representing parallel iname SAME condition in sio
    conc_inames, _ = partition_inames_by_concurrency(knl["loopy_kernel"])
    par_iname_condition = " and ".join(
        "{0} = {0}'".format(iname) for iname in conc_inames)

    # {{{ Intra-thread relationship between stmt_a and stmt_b

    sio_intra_thread_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii', j', jj'] -> [{0}=1, i, ii, j, jj] : "
        "0 <= i,ii,i',ii' < pi and 0 <= j,jj,j',jj' < pj "
        "and (ii > ii' or (ii = ii' and jj >= jj')) "
        "and {1} "
        "}}".format(
            STATEMENT_VAR_NAME,
            par_iname_condition,
            )
        )

    _check_orderings_for_stmt_pair(
        "stmt_a", "stmt_b", pworders,
        sio_intra_thread_exp=sio_intra_thread_exp)

    # }}}

    # {{{ Relationship between gbar and stmt_a

    # intra-thread case
    sio_intra_thread_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii'] -> [{0}=1, i, ii, j, jj] : "
        "0 <= i,ii,i',ii' < pi and 0 <= j,jj < pj "  # domains
        "and i = i' "  # parallel inames must be same
        "and ii >= ii' "  # before->after condtion
        "}}".format(
            STATEMENT_VAR_NAME,
            )
        )

    # intra-group case
    # (this test also confirms that our SIO construction accounts for the fact
    # that global barriers *also* syncronize across threads *within* a group,
    # which is why the before->after condition below is *not*
    # "and (ii > ii' or (ii = ii' and jj > 0))")
    sio_intra_group_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii'] -> [{0}=1, i, ii, j, jj] : "
        "0 <= i,ii,i',ii' < pi and 0 <= j,jj < pj "  # domains
        "and i = i' "  # GID inames must be same
        "and ii >= ii'"  # before->after condtion
        "}}".format(
            STATEMENT_VAR_NAME,
            )
        )

    # global case
    sio_global_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii'] -> [{0}=1, i, ii, j, jj] : "
        "0 <= i,ii,i',ii' < pi and 0 <= j,jj < pj "  # domains
        "and ii >= ii' "  # before->after condtion
        "}}".format(
            STATEMENT_VAR_NAME,
            )
        )

    _check_orderings_for_stmt_pair(
        "gbar", "stmt_a", pworders,
        sio_intra_thread_exp=sio_intra_thread_exp,
        # sio_intra_group_exp=sio_intra_group_exp,
        sio_global_exp=sio_global_exp)

    # }}}

    # {{{ Relationship between stmt_b and lbar1

    # intra thread case

    sio_intra_thread_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii', j', jj'] -> [{0}=1, i, ii, j, jj] : "
        "0 <= i,ii,i',ii' < pi and 0 <= j,jj,j',jj' < pj "  # domains
        "and i = i' and j = j'"  # parallel inames must be same
        "and (ii > ii' or (ii = ii' and jj >= jj'))"  # before->after condtion
        "}}".format(
            STATEMENT_VAR_NAME,
            )
        )

    # intra-group case

    sio_intra_group_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii', j', jj'] -> [{0}=1, i, ii, j, jj] : "
        "0 <= i,ii,i',ii' < pi and 0 <= j,jj,j',jj' < pj "  # domains
        "and i = i' "  # GID parallel inames must be same
        "and (ii > ii' or (ii = ii' and jj >= jj'))"  # before->after condtion
        "}}".format(
            STATEMENT_VAR_NAME,
            )
        )

    # global case

    sio_global_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii', j', jj'] -> [{0}=1, i, ii, j, jj] : "
        "0 <= i,ii,i',ii' < pi and 0 <= j,jj,j',jj' < pj "  # domains
        "and ii > ii'"  # before->after condtion
        "}}".format(
            STATEMENT_VAR_NAME,
            )
        )

    _check_orderings_for_stmt_pair(
        "stmt_b", "lbar1", pworders,
        sio_intra_thread_exp=sio_intra_thread_exp,
        sio_intra_group_exp=sio_intra_group_exp,
        sio_global_exp=sio_global_exp,
        )

    # }}}

    # {{{ Relationship between stmt_a and stmt_b

    # intra thread case

    sio_intra_thread_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii', j', jj'] -> [{0}=1, i, ii, j, jj] : "
        "0 <= i,ii,i',ii' < pi and 0 <= j,jj,j',jj' < pj "  # domains
        "and i = i' and j = j'"  # parallel inames must be same
        "and (ii > ii' or (ii = ii' and jj >= jj'))"  # before->after condtion
        "}}".format(
            STATEMENT_VAR_NAME,
            )
        )

    # intra-group case

    sio_intra_group_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii', j', jj'] -> [{0}=1, i, ii, j, jj] : "
        "0 <= i,ii,i',ii' < pi and 0 <= j,jj,j',jj' < pj "  # domains
        "and i = i' "  # GID parallel inames must be same
        "and (ii > ii' or (ii = ii' and jj >= jj'))"  # before->after condtion
        "}}".format(
            STATEMENT_VAR_NAME,
            )
        )

    _check_orderings_for_stmt_pair(
        "stmt_a", "stmt_b", pworders,
        sio_intra_thread_exp=sio_intra_thread_exp,
        sio_intra_group_exp=sio_intra_group_exp,
        )

    # }}}

    # {{{ Relationship between lbar1 and stmt_c

    # intra thread case

    sio_intra_thread_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii', j', jj'] -> [{0}=1] : "
        "0 <= i',ii' < pi and 0 <= j',jj' < pj "  # domains
        "}}".format(
            STATEMENT_VAR_NAME,
            )
        )

    # intra-group case

    sio_intra_group_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii', j', jj'] -> [{0}=1] : "
        "0 <= i',ii' < pi and 0 <= j',jj' < pj "  # domains
        "}}".format(
            STATEMENT_VAR_NAME,
            )
        )

    # global case

    # (only happens before if not last iteration of ii
    sio_global_exp = _isl_map_with_marked_dims(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', ii', j', jj'] -> [{0}=1] : "
        "0 <= i',ii' < pi and 0 <= j',jj' < pj "  # domains
        "and ii' < pi-1"
        "}}".format(
            STATEMENT_VAR_NAME,
            )
        )

    _check_orderings_for_stmt_pair(
        "lbar1", "stmt_c", pworders,
        sio_intra_thread_exp=sio_intra_thread_exp,
        sio_intra_group_exp=sio_intra_group_exp,
        sio_global_exp=sio_global_exp,
        )

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
        assumptions=assumptions,
        lang_version=(2018, 2)
        )
    knl = lp.tag_inames(knl, {"l0": "l.0", "l1": "l.1", "g0": "g.0"})

    # Get a linearization
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    stmt_id_pairs = [("stmt_j1", "stmt_2"), ("stmt_1", "stmt_i0")]
    pworders = get_pairwise_statement_orderings(
        lin_knl, lin_items, stmt_id_pairs, perform_closure_checks=True)

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

    # (this test also confirms that our sched/SIO construction accounts for the
    # fact that global barriers *also* syncronize across threads *within* a
    # group, which is why dim 2 below is asigned the value 3 instead of 2)
    sched_stmt_j1_intra_group_exp = isl.Map(
        "[ij_start, ij_end, lg_end] -> {"
        "[%s=0, i, j, l0, l1, g0] -> [%s] : "
        "%s and %s}"  # iname bounds
        % (
            STATEMENT_VAR_NAME,
            _lex_point_string(
                ["2", "i", "3", "j", "1"],  # lex points
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
        lin_knl, lin_items, stmt_id_pairs, perform_closure_checks=True)

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
    from loopy.schedule import RunInstruction
    all_stmt_ids = [
        lin_item.insn_id for lin_item in lin_items
        if isinstance(lin_item, RunInstruction)]
    from itertools import product
    stmt_id_pairs = []
    for idx, sid in enumerate(all_stmt_ids):
        stmt_id_pairs.extend(product([sid], all_stmt_ids[idx+1:]))

    # Generate pairwise ordering info for every pair
    get_pairwise_statement_orderings(
        lin_knl, lin_items, stmt_id_pairs, perform_closure_checks=True)

# }}}


# {{{ test_blex_map_transitivity_with_triangular_domain

def test_blex_map_transitivity_with_triangular_domain():

    assumptions = "i_start + 1 <= ijk_end"
    knl = lp.make_kernel(
        [
            "{[i,j,k]: i_start<=i<ijk_end and i<=j<ijk_end and i+j<=k<ijk_end}",
        ],
        """
        for i
            <>temp0 = 0  {id=stmt_i0}
            ... lbarrier  {id=stmt_b0,dep=stmt_i0}
            <>temp1 = 1  {id=stmt_i1,dep=stmt_b0}
            for j
                <>tempj0 = 0  {id=stmt_j0,dep=stmt_i1}
                ... lbarrier {id=stmt_jb0,dep=stmt_j0}
                ... gbarrier {id=stmt_jbb0,dep=stmt_j0}
                <>tempj1 = 0  {id=stmt_j1,dep=stmt_jb0}
                <>tempj2 = 0  {id=stmt_j2,dep=stmt_j1}
                for k
                    <>tempk0 = 0  {id=stmt_k0,dep=stmt_j2}
                    ... lbarrier {id=stmt_kb0,dep=stmt_k0}
                    <>tempk1 = 0  {id=stmt_k1,dep=stmt_kb0}
                end
            end
            <>temp2 = 0  {id=stmt_i2,dep=stmt_j0}
        end
        """,
        assumptions=assumptions,
        lang_version=(2018, 2)
        )

    ref_knl = knl

    # Get a linearization
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    stmt_id_pairs = [
        ("stmt_i0", "stmt_i1"),
        ("stmt_i1", "stmt_j0"),
        ("stmt_j0", "stmt_j1"),
        ("stmt_j1", "stmt_j2"),
        ("stmt_j2", "stmt_k0"),
        ("stmt_k0", "stmt_k1"),
        ("stmt_k1", "stmt_i2"),
        ]
    # Set perform_closure_checks=True and get the orderings
    get_pairwise_statement_orderings(
        lin_knl, lin_items, stmt_id_pairs, perform_closure_checks=True)

    # Now try it with concurrent i loop
    knl = lp.tag_inames(knl, "i:g.0")

    # Get a linearization
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    stmt_id_pairs = [
        ("stmt_i0", "stmt_i1"),
        ("stmt_i1", "stmt_j0"),
        ("stmt_j0", "stmt_j1"),
        ("stmt_j1", "stmt_j2"),
        ("stmt_j2", "stmt_k0"),
        ("stmt_k0", "stmt_k1"),
        ("stmt_k1", "stmt_i2"),
        ]
    # Set perform_closure_checks=True and get the orderings
    get_pairwise_statement_orderings(
        lin_knl, lin_items, stmt_id_pairs, perform_closure_checks=True)

    # Now try it with concurrent i and j loops
    knl = lp.tag_inames(knl, "j:g.1")

    # Get a linearization
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    stmt_id_pairs = [
        ("stmt_i0", "stmt_i1"),
        ("stmt_i1", "stmt_j0"),
        ("stmt_j0", "stmt_j1"),
        ("stmt_j1", "stmt_j2"),
        ("stmt_j2", "stmt_k0"),
        ("stmt_k0", "stmt_k1"),
        ("stmt_k1", "stmt_i2"),
        ]
    # Set perform_closure_checks=True and get the orderings
    get_pairwise_statement_orderings(
        lin_knl, lin_items, stmt_id_pairs, perform_closure_checks=True)

    # Now try it with concurrent i and k loops
    knl = ref_knl
    knl = lp.tag_inames(knl, {"i": "g.0", "k": "g.1"})

    # Get a linearization
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    stmt_id_pairs = [
        ("stmt_i0", "stmt_i1"),
        ("stmt_i1", "stmt_j0"),
        ("stmt_j0", "stmt_j1"),
        ("stmt_j1", "stmt_j2"),
        ("stmt_j2", "stmt_k0"),
        ("stmt_k0", "stmt_k1"),
        ("stmt_k1", "stmt_i2"),
        ]
    # Set perform_closure_checks=True and get the orderings
    get_pairwise_statement_orderings(
        lin_knl, lin_items, stmt_id_pairs, perform_closure_checks=True)

    # FIXME create some expected sios and compare

# }}}


# {{{ test_blex_map_transitivity_with_duplicate_conc_inames

def test_blex_map_transitivity_with_duplicate_conc_inames():

    knl = lp.make_kernel(
        [
            "{[i,j,ii,jj]: 0 <= i,j,jj < n and i <= ii < n}",
            "{[k, kk]: 0 <= k,kk < n}",
        ],
        """
        for i
            for ii
                <> si = 0  {id=si}
                ... lbarrier {id=bari, dep=si}
            end
        end
        for j
            for jj
                <> sj = 0  {id=sj, dep=si}
                ... lbarrier {id=barj, dep=sj}
            end
        end
        for k
            for kk
                <> sk = 0  {id=sk, dep=sj}
                ... lbarrier {id=bark, dep=sk}
            end
        end
        """,
        assumptions="0 < n",
        lang_version=(2018, 2)
        )

    knl = lp.tag_inames(knl, {"i": "l.0", "j": "l.0", "k": "l.0"})

    # Get a linearization
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    stmt_id_pairs = [
        ("si", "si"),
        ("si", "sj"),
        ("si", "sk"),
        ("sj", "sj"),
        ("sj", "sk"),
        ("sk", "sk"),
        ]

    # Set perform_closure_checks=True and get the orderings
    get_pairwise_statement_orderings(
        lin_knl, lin_items, stmt_id_pairs, perform_closure_checks=True)

    # print(prettier_map_string(pw_sios[("si", "sj")].sio_intra_thread))
    # print(prettier_map_string(pw_sios[("si", "sj")].sio_intra_group))
    # print(prettier_map_string(pw_sios[("si", "sj")].sio_global))

    # FIXME create some expected sios and compare

# }}}


# {{{ Dependency tests

# {{{ Dependency creation and checking (without transformations)

# {{{ test_add_dependency_with_new_deps

def test_add_dependency_with_new_deps():
    """Use add_dependency to add new deps to kernels and make sure that the
    correct dep is being added to the correct instruction. Also make sure that
    these deps can be succesfully checked for violation. Also, while we're
    here, test to make sure make_dep_map() produces the correct result."""

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
        lang_version=(2018, 2)
        )
    knl = lp.add_and_infer_dtypes(
            knl, {"a": np.float32, "b": np.float32, "c": np.float32})

    for stmt in knl["loopy_kernel"].instructions:
        assert not stmt.dependencies

    # Add a dependency to stmt_b
    dep_b_on_a = make_dep_map(
        "[pi] -> {{ [i'] -> [i] : i > i' "
        "and {0} "
        "}}".format(assumptions_str),
        knl_with_domains=knl["loopy_kernel"])
    knl = lp.add_dependency(knl, "id:stmt_b", ("id:stmt_a", dep_b_on_a))

    # Compare deps to expected deps (but don't check for satisfaction yet)
    _compare_dependencies(
        knl,
        {"stmt_b": {
            "stmt_a": [dep_b_on_a, ]}})

    # {{{ Test make_dep_map while we're here

    dep_b_on_a_test = _isl_map_with_marked_dims(
        "[pi] -> {{ [{3}'=0, i'] -> [{3}=1, i] : i > i' "
        "and {0} and {1} and {2} "
        "}}".format(
            i_range_str,
            i_range_str_p,
            assumptions_str,
            STATEMENT_VAR_NAME,
            ))
    _align_and_compare_maps([(dep_b_on_a, dep_b_on_a_test)])

    # }}}

    # Add a second dependency to stmt_b
    dep_b_on_a_2 = make_dep_map(
        "[pi] -> {{ [i'] -> [i] : i = i' "
        "and {0}"
        "}}".format(assumptions_str),
        knl_with_domains=knl["loopy_kernel"])
    knl = lp.add_dependency(knl, "id:stmt_b", ("id:stmt_a", dep_b_on_a_2))

    # Compare deps to expected deps (but don't check for satisfaction yet)
    _compare_dependencies(
        knl,
        {"stmt_b": {
            "stmt_a": [dep_b_on_a, dep_b_on_a_2]}})

    # {{{ Test make_dep_map while we're here

    dep_b_on_a_2_test = _isl_map_with_marked_dims(
        "[pi] -> {{ [{3}'=0, i'] -> [{3}=1, i] : i = i' "
        "and {0} and {1} and {2} }}".format(
            i_range_str,
            i_range_str_p,
            assumptions_str,
            STATEMENT_VAR_NAME,
            ))
    _align_and_compare_maps([(dep_b_on_a_2, dep_b_on_a_2_test)])

    # }}}

    # Add dependencies to stmt_c

    dep_c_on_a = make_dep_map(
        "[pi] -> {{ [i'] -> [i] : i >= i' "
        "and {0} "
        "}}".format(assumptions_str),
        knl_with_domains=knl["loopy_kernel"])

    dep_c_on_b = make_dep_map(
        "[pi] -> {{ [i'] -> [i] : i >= i' "
        "and {0} "
        "}}".format(assumptions_str),
        knl_with_domains=knl["loopy_kernel"])

    knl = lp.add_dependency(knl, "id:stmt_c", ("id:stmt_a", dep_c_on_a))
    knl = lp.add_dependency(knl, "id:stmt_c", ("id:stmt_b", dep_c_on_b))

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


# {{{ test_make_dep_map

def test_make_dep_map():
    """Make sure make_dep_map() produces the desired result. This is also
    tested inside other test functions, but here we specifically test cases
    where the statement inames don't match."""

    # Make kernel and use OLD deps to control linearization order for now
    i_range_str = "0 <= i < n"
    i_range_str_p = "0 <= i' < n"
    j_range_str = "0 <= j < n"
    j_range_str_p = "0 <= j' < n"
    k_range_str = "0 <= k < n"
    knl = lp.make_kernel(
        "{[i,j,k]: %s}" % (" and ".join([i_range_str, j_range_str, k_range_str])),
        """
        a[i,j] = 3.14  {id=stmt_a}
        b[k] = a[i,k]  {id=stmt_b, dep=stmt_a}
        """,
        lang_version=(2018, 2)
        )
    knl = lp.add_and_infer_dtypes(knl, {"a,b": np.float32})

    for stmt in knl["loopy_kernel"].instructions:
        assert not stmt.dependencies

    # Add a dependency to stmt_b
    dep_b_on_a = make_dep_map(
        "[n] -> { [i',j'] -> [i,k] : i > i' and j' < k}",
        knl_with_domains=knl["loopy_kernel"])

    # Create expected dep
    dep_b_on_a_test = _isl_map_with_marked_dims(
        "[n] -> {{ [{0}'=0, i', j'] -> [{0}=1, i, k] : i > i' and j' < k"
        " and {1} }}".format(
            STATEMENT_VAR_NAME,
            " and ".join([
                i_range_str,
                i_range_str_p,
                j_range_str_p,
                k_range_str,
                ])
            ))
    _align_and_compare_maps([(dep_b_on_a, dep_b_on_a_test)])

# }}}


# {{{ test_new_dependencies_finite_diff:

def test_new_dependencies_finite_diff():
    """Test find_unsatisfied_dependencies() using several variants of a finite
    difference kernel, some of which violate dependencies."""

    # Define kernel
    knl = lp.make_kernel(
        "[nx,nt] -> {[x, t]: 0<=x<nx and 0<=t<nt}",
        "u[t+2,x+1] = 2*u[t+1,x+1] + dt**2/dx**2 "
        "* (u[t+1,x+2] - 2*u[t+1,x+1] + u[t+1,x]) - u[t,x+1]  {id=stmt}")
    knl = lp.add_dtypes(
        knl, {"u": np.float32, "dx": np.float32, "dt": np.float32})

    # Define and add dependency
    dep = make_dep_map(
        "[nx,nt] -> { [x', t'] -> [x, t] : "
        "((x = x' and t = t'+2) or "
        " (x'-1 <= x <= x'+1 and t = t' + 1)) }",
        self_dep=True, knl_with_domains=knl["loopy_kernel"])
    knl = lp.add_dependency(knl, "id:stmt", ("id:stmt", dep))

    # {{{ Test make_dep_map while we're here

    dep_test = make_dep_map(
        "[nx,nt] -> { [x', t'] -> [x, t] : "
        "((x = x' and t = t'+2) or "
        " (x'-1 <= x <= x'+1 and t = t' + 1)) and "
        "0 <= x < nx and 0 <= t < nt and "
        "0 <= x' < nx and 0 <= t' < nt }",
        self_dep=True)

    _align_and_compare_maps([(dep, dep_test)])

    # }}}

    ref_knl = knl

    # {{{ Test find_unsatisfied_dependencies with corrct loop nest order

    # Prioritize loops correctly
    knl = lp.prioritize_loops(knl, "t,x")

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmt": {"stmt": [dep, ]}, },
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}

    # {{{ Test find_unsatisfied_dependencies with incorrect loop nest order

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

    # {{{ Test find_unsatisfied_dependencies with parallel x and no barrier

    # Parallelize the x loop
    knl = ref_knl
    knl = lp.prioritize_loops(knl, "t,x")
    knl = lp.tag_inames(knl, "x:l.0")

    # Make sure unsatisfied deps are caught
    lin_items, proc_knl, lin_knl = _process_and_linearize(knl)

    # Without a barrier, deps are not satisfied
    # Make sure there is no barrier, and that unsatisfied deps are caught
    from loopy.schedule import Barrier
    for lin_item in lin_items:
        assert not isinstance(lin_item, Barrier)

    unsatisfied_deps = lp.find_unsatisfied_dependencies(
        proc_knl, lin_items, stop_on_first_violation=False)

    assert len(unsatisfied_deps) == 1

    # }}}

    # {{{ Test find_unsatisfied_dependencies with parallel x and included barrier

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

    knl = lp.add_dependency(knl, "id:stmt", ("id:stmt", dep))

    knl = lp.prioritize_loops(knl, "t,x")
    knl = lp.tag_inames(knl, "x:l.0")

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmt": {"stmt": [dep, ]}, },
        return_unsatisfied=True)

    assert not unsatisfied_deps

    # }}}

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
    knl = lp.add_dependency(knl, "id:stmt1", ("id:stmt0", deepcopy(dep_orig)))
    knl = lp.add_dependency(knl, "id:stmt2", ("id:stmt1", deepcopy(dep_orig)))

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
    knl = lp.add_dependency(knl, "id:stmt1", ("id:stmt0", deepcopy(dep_le)))
    knl = lp.add_dependency(
        knl, "id:stmt2 or id:stmt3 or id:stmt4",
        ("id:stmt1", deepcopy(dep_eq)))

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

    knl = lp.add_dependency(knl, "id:stmt1", ("id:stmt0", deepcopy(dep_le)))
    knl = lp.add_dependency(
        knl, "id:stmt2 or id:stmt3 or id:stmt4 or id:stmt5",
        ("id:stmt1", deepcopy(dep_eq)))

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

    # test case where subst def is removed, has deps, and
    # inames of subst_def don't match subst usage

    knl = lp.make_kernel(
        "{[i,j,k,m]: 0 <= i,j,k,m < n}",
        """
        for i,j
            <>temp0 = 0.1*i {id=stmt0}
        end
        for k
            <>tsq = temp0**2  {id=stmt1,dep=stmt0}
        end
        for m
            <>res = 23*tsq + 25*tsq  {id=stmt2,dep=stmt1}
        end
        """)
    knl = lp.add_and_infer_dtypes(knl, {"temp0,tsq,res": np.float32})

    dep_1_on_0 = make_dep_map(
        "[n] -> { [i', j']->[k] : 0 <= i',j',k < n }", self_dep=False)
    dep_2_on_1 = make_dep_map(
        "[n] -> { [k']->[m] : 0 <= k',m < n }", self_dep=False)

    from copy import deepcopy
    knl = lp.add_dependency(knl, "id:stmt1", ("id:stmt0", deepcopy(dep_1_on_0)))
    knl = lp.add_dependency(knl, "id:stmt2", ("id:stmt1", deepcopy(dep_2_on_1)))

    knl = lp.assignment_to_subst(knl, "tsq")

    dep_exp = make_dep_map(
        "[n] -> { [i', j']->[m] : 0 <= i',j',m < n }", self_dep=False)

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {
            "stmt2": {"stmt0": [dep_exp, ]},
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
    knl = lp.add_dependency(knl, "id:stmtc", ("id:stmtb", dep_eq))

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


# {{{ test_rename_inames_with_dependencies

def test_rename_inames_with_dependencies():
    # When rename_iname is called and the new iname
    # *doesn't* already exist, then duplicate_inames is called,
    # and we test that elsewhere. Here we test the case where
    # rename_iname is called and the new iname already exists.

    knl = lp.make_kernel(
        "{[i,j,m,j_new]: 0 <= i,j,m,j_new < n}",
        """
        b[i,j] = a[i,j]  {id=stmtb}
        c[i,j] = a[i,j]  {id=stmtc,dep=stmtb}
        e[i,j_new] = 1.1
        d[m] = 5.5  {id=stmtd,dep=stmtc}
        """)
    knl = lp.add_and_infer_dtypes(knl, {"a,d": np.float32})

    dep_c_on_b = make_dep_map(
        "[n] -> { [i', j']->[i, j] : 0 <= i,i',j,j' < n and i' = i and j' = j }",
        self_dep=False)
    dep_c_on_c = make_dep_map(
        "[n] -> { [i', j']->[i, j] : 0 <= i,i',j,j' < n and i' < i and j' < j }",
        self_dep=True)
    dep_d_on_c = make_dep_map(
        "[n] -> { [i', j']->[m] : 0 <= m,i',j' < n }",
        self_dep=False)

    # Create dep stmtb->stmtc
    knl = lp.add_dependency(knl, "id:stmtc", ("id:stmtb", dep_c_on_b))
    knl = lp.add_dependency(knl, "id:stmtc", ("id:stmtc", dep_c_on_c))
    knl = lp.add_dependency(knl, "id:stmtd", ("id:stmtc", dep_d_on_c))

    # Rename j within stmtc

    knl = lp.rename_iname(
        knl, "j", "j_new", within="id:stmtc", existing_ok=True)

    dep_c_on_b_exp = make_dep_map(
        "[n] -> { [i', j']->[i, j_new] : "
        "0 <= i,i',j_new,j' < n and i' = i and j' = j_new}",
        self_dep=False)
    dep_c_on_c_exp = make_dep_map(
        "[n] -> { [i', j_new']->[i, j_new] : "
        "0 <= i,i',j_new,j_new' < n and i' < i and j_new' < j_new }",
        self_dep=True)
    dep_d_on_c_exp = make_dep_map(
        "[n] -> { [i', j_new']->[m] : 0 <= m,i',j_new' < n }",
        self_dep=False)

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {
            "stmtc": {"stmtb": [dep_c_on_b_exp, ], "stmtc": [dep_c_on_c_exp, ]},
            "stmtd": {"stmtc": [dep_d_on_c_exp, ]},
        },
        return_unsatisfied=True)

    assert not unsatisfied_deps

# }}}


# {{{ test_split_iname_with_dependencies

def test_split_iname_with_dependencies():
    knl = lp.make_kernel(
        "{[i]: 0<=i<p}",
        """
        a[i] = 0.1  {id=stmt0}
        b[i] = a[i]  {id=stmt1,dep=stmt0}
        """,
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

    knl = lp.add_dependency(knl, "id:stmt1", ("id:stmt0", dep_satisfied))
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

    knl = lp.add_dependency(knl, "id:stmt1", ("id:stmt0", dep_satisfied))
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

    knl = lp.add_dependency(knl, "id:stmt1", ("id:stmt0", dep_satisfied))
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

    knl = lp.add_dependency(knl, "id:stmt1", ("id:stmt0", dep_unsatisfied))
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

    knl = lp.add_dependency(knl, "id:stmt1", ("id:stmt0", dep_stmt1_on_stmt0_eq))
    knl = lp.add_dependency(knl, "id:stmt1", ("id:stmt0", dep_stmt1_on_stmt0_lt))
    knl = lp.add_dependency(knl, "id:stmt3", ("id:stmt2", dep_stmt3_on_stmt2_eq))

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

    knl = lp.add_dependency(
        knl, "id:stmtc", ("id:stmta", dep_c_on_a))

    # Intentionally make order of x and t different from transform_map below
    # to test alignment steps in map_domain
    dep_e_on_c = _isl_map_with_marked_dims(
        "[nx, nt, ni] -> {{"
        "[{0}' = 0, t', x'] -> [{0} = 1, i] : "
        "0 <= x' < nx and 0 <= t' < nt and 0 <= i < ni"
        "}}".format(STATEMENT_VAR_NAME))

    knl = lp.add_dependency(
        knl, "id:stmte", ("id:stmtc", dep_e_on_c))

    # }}}

    # {{{ Apply domain change mapping

    # Create map_domain mapping:
    import islpy as isl
    transform_map = isl.BasicMap(
        "[nt] -> {[t] -> [t_outer, t_inner]: "
        "0 <= t_inner < 32 and "
        "32*t_outer + t_inner = t and "
        "0 <= 32*t_outer + t_inner < nt}")

    # Call map_domain to transform kernel
    knl = lp.map_domain(knl, transform_map)

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


# {{{ test_map_domain_with_inames_missing_in_transform_map

def test_map_domain_with_inames_missing_in_transform_map():

    # Make sure map_domain updates deps correctly when the mapping doesn't
    # include all the dims in the domain.

    # {{{ Make kernel

    knl = lp.make_kernel(
        "[nx,nt] -> {[x, y, z, t]: 0 <= x,y,z < nx and 0 <= t < nt}",
        """
        a[y,x,t,z] = b[y,x,t,z]  {id=stmta}
        """,
        lang_version=(2018, 2),
        )
    knl = lp.add_and_infer_dtypes(knl, {"b": np.float32})

    # }}}

    # {{{ Create dependency

    dep = _isl_map_with_marked_dims(
        "[nx, nt] -> {{"
        "[{0}' = 0, x', y', z', t'] -> [{0} = 0, x, y, z, t] : "
        "0 <= x,y,z,x',y',z' < nx and 0 <= t,t' < nt and "
        "t' < t and x' < x and y' < y and z' < z"
        "}}".format(STATEMENT_VAR_NAME))

    knl = lp.add_dependency(knl, "id:stmta", ("id:stmta", dep))

    # }}}

    # {{{ Apply domain change mapping

    # Create map_domain mapping that only includes t and y
    # (x and z should be unaffected)
    transform_map = isl.BasicMap(
        "[nx,nt] -> {[t, y] -> [t_outer, t_inner, y_new]: "
        "0 <= t_inner < 32 and "
        "32*t_outer + t_inner = t and "
        "0 <= 32*t_outer + t_inner < nt and "
        "y = y_new"
        "}")

    # Call map_domain to transform kernel
    knl = lp.map_domain(knl, transform_map)

    # }}}

    # {{{ Create expected dependency after transformation

    dep_exp = _isl_map_with_marked_dims(
        "[nx, nt] -> {{"
        "[{0}' = 0, x', y_new', z', t_outer', t_inner'] -> "
        "[{0} = 0, x, y_new, z, t_outer, t_inner] : "
        "0 <= x,z,x',z' < nx "  # old bounds
        "and 0 <= t_inner,t_inner' < 32 and 0 <= y_new,y_new' < nx "  # new bounds
        "and 0 <= 32*t_outer + t_inner < nt "  # new bounds
        "and 0 <= 32*t_outer' + t_inner' < nt "  # new bounds
        "and x' < x and z' < z "  # old constraints
        "and y_new' < y_new "  # new constraint
        "and 32*t_outer' + t_inner' < 32*t_outer + t_inner"  # new constraint
        "}}".format(STATEMENT_VAR_NAME))

    # }}}

    # {{{ Make sure deps are correct and satisfied

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmta": {"stmta": [dep_exp, ]}},
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

    knl = lp.add_dependency(
        knl, "id:"+stmt_after, ("id:"+stmt_before, dep_map))

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
    # Insert 'statement' dim into transform map
    transform_map = insert_and_name_isl_dims(
            transform_map, dim_type.in_, [STATEMENT_VAR_NAME+BEFORE_MARK], 0)
    transform_map = insert_and_name_isl_dims(
            transform_map, dim_type.out, [STATEMENT_VAR_NAME], 0)
    # Add stmt = stmt' constraint
    transform_map = add_eq_isl_constraint_from_names(
        transform_map, STATEMENT_VAR_NAME, STATEMENT_VAR_NAME+BEFORE_MARK)

    # Apply transform map to dependency
    mapped_dep_map = dep_map.apply_range(transform_map).apply_domain(transform_map)
    mapped_dep_map = append_mark_to_isl_map_var_names(
        mapped_dep_map, dim_type.in_, BEFORE_MARK)

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


# {{{ test_add_prefetch_with_dependencies

# FIXME handle deps during prefetch

'''

def test_add_prefetch_with_dependencies():

    lp.set_caching_enabled(False)
    knl = lp.make_kernel(
        "[p] -> { [i,j,k,m] : 0 <= i,j < p and 0 <= k,m < 16}",
        """
        for i,j,k,m
            a[i+1,j+1,k+1,m+1] = a[i,j,k,m]  {id=stmt}
        end
        """,
        assumptions="p >= 1",
        lang_version=(2018, 2)
        )
    knl = lp.add_and_infer_dtypes(knl, {"a": np.float32})

    dep_init = make_dep_map(
        "{ [i',j',k',m'] -> [i,j,k,m] : "
        "i' + 1 = i and j' + 1 = j and k' + 1 = k and m' + 1 = m }",
        self_dep=True, knl_with_domains=knl)
    knl = lp.add_dependency(knl, "id:stmt", ("id:stmt", dep_init))

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {"stmt": {"stmt": [dep_init, ]}},
        return_unsatisfied=True)

    assert not unsatisfied_deps

    knl = lp.add_prefetch(
        knl, "a", sweep_inames=["k", "m"],
        fetch_outer_inames=frozenset({"i", "j"}),
        # dim_arg_names=["k_fetch", "m_fetch"],  # TODO not sure why these don't work
        )

    # create expected deps
    dep_stmt_on_fetch_exp = make_dep_map(
        "{ [i',j',a_dim_2',a_dim_3'] -> [i,j,k,m] : "
        "i' = i and j' = j }",
        knl_with_domains=knl)
    dep_fetch_on_stmt_exp = make_dep_map(
        "{ [i',j',k',m'] -> [i,j,a_dim_2,a_dim_3] : "
        "i' + 1 = i and j' + 1 = j "
        "and 0 <= k',m' < 15 "
        "}",
        knl_with_domains=knl)
    # (make_dep_map will set k',m' upper bound to 16, so add manually^)

    # Why is this necessary to avoid dependency cycle?
    knl.id_to_insn["a_fetch_rule"].depends_on_is_final = True

    # Compare deps and make sure they are satisfied
    unsatisfied_deps = _compare_dependencies(
        knl,
        {
            "stmt": {"stmt": [dep_init], "a_fetch_rule": [dep_stmt_on_fetch_exp]},
            "a_fetch_rule": {"stmt": [dep_fetch_on_stmt_exp]},
        },
        return_unsatisfied=True)

    assert not unsatisfied_deps

'''

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

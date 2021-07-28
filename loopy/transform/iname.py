__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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


import islpy as isl
from islpy import dim_type

from loopy.symbolic import (
        RuleAwareIdentityMapper, RuleAwareSubstitutionMapper,
        SubstitutionRuleMappingContext)
from loopy.diagnostic import LoopyError
from pytools import Record

from loopy.translation_unit import (TranslationUnit,
                                    for_each_kernel)
from loopy.kernel import LoopKernel
from loopy.kernel.function_interface import CallableKernel


__doc__ = """
.. currentmodule:: loopy

.. autofunction:: split_iname

.. autofunction:: chunk_iname

.. autofunction:: join_inames

.. autofunction:: untag_inames

.. autofunction:: tag_inames

.. autofunction:: duplicate_inames

.. autofunction:: get_iname_duplication_options

.. autofunction:: has_schedulable_iname_nesting

.. autofunction:: prioritize_loops

.. autofunction:: rename_iname

.. autofunction:: remove_unused_inames

.. autofunction:: split_reduction_inward

.. autofunction:: split_reduction_outward

.. autofunction:: affine_map_inames

.. autofunction:: find_unused_axis_tag

.. autofunction:: make_reduction_inames_unique

.. autofunction:: add_inames_to_insn

.. autofunction:: add_inames_for_unused_hw_axes

"""


# {{{ set loop priority

@for_each_kernel
def set_loop_priority(kernel, loop_priority):
    from warnings import warn
    warn("set_loop_priority is deprecated. Use prioritize_loops instead. "
         "Attention: A call to set_loop_priority will overwrite any previously "
         "set priorities!", DeprecationWarning, stacklevel=2)

    if isinstance(loop_priority, str):
        loop_priority = tuple(s.strip()
                              for s in loop_priority.split(",") if s.strip())
    loop_priority = tuple(loop_priority)

    return kernel.copy(loop_priority=frozenset([loop_priority]))


@for_each_kernel
def prioritize_loops(kernel, loop_priority):
    """Indicates the textual order in which loops should be entered in the
    kernel code. Note that this priority has an advisory role only. If the
    kernel logically requires a different nesting, priority is ignored.
    Priority is only considered if loop nesting is ambiguous.

    prioritize_loops can be used multiple times. If you do so, each given
    *loop_priority* specifies a scheduling constraint. The constraints from
    all calls to prioritize_loops together establish a partial order on the
    inames (see https://en.wikipedia.org/wiki/Partially_ordered_set).

    :arg: an iterable of inames, or, for brevity, a comma-separated string of
        inames
    """

    assert isinstance(kernel, LoopKernel)
    if isinstance(loop_priority, str):
        loop_priority = tuple(s.strip()
                              for s in loop_priority.split(",") if s.strip())
    loop_priority = tuple(loop_priority)

    return kernel.copy(loop_priority=kernel.loop_priority.union([loop_priority]))

# }}}


# {{{ Handle loop nest constraints

# {{{ Classes to house loop nest constraints

# {{{ UnexpandedInameSet

class UnexpandedInameSet(Record):
    def __init__(self, inames, complement=False):
        Record.__init__(
            self,
            inames=inames,
            complement=complement,
            )

    def contains(self, inames):
        if isinstance(inames, set):
            return (not (inames & self.inames) if self.complement
                else inames.issubset(self.inames))
        else:
            return (inames not in self.inames if self.complement
                else inames in self.inames)

    def get_inames_represented(self, iname_universe=None):
        """Return the set of inames represented by the UnexpandedInameSet
        """
        if self.complement:
            if not iname_universe:
                raise ValueError(
                    "Cannot expand UnexpandedInameSet %s without "
                    "iname_universe." % (self))
            return iname_universe-self.inames
        else:
            return self.inames.copy()

    def __lt__(self, other):
        # FIXME is this function really necessary? (for caching?)
        # If so, what should it return?
        return self.__hash__() < other.__hash__()

    def __hash__(self):
        return hash(repr(self))

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        key_builder.rec(key_hash, self.inames)
        key_builder.rec(key_hash, self.complement)

    def __str__(self):
        return "%s{%s}" % ("~" if self.complement else "",
            ",".join(i for i in sorted(self.inames)))

# }}}


# {{{ LoopNestConstraints

class LoopNestConstraints(Record):
    def __init__(self, must_nest=None, must_not_nest=None,
                 must_nest_graph=None):
        Record.__init__(
            self,
            must_nest=must_nest,
            must_not_nest=must_not_nest,
            must_nest_graph=must_nest_graph,
            )

    def __hash__(self):
        return hash(repr(self))

    def update_persistent_hash(self, key_hash, key_builder):
        """Custom hash computation function for use with
        :class:`pytools.persistent_dict.PersistentDict`.
        """

        key_builder.rec(key_hash, self.must_nest)
        key_builder.rec(key_hash, self.must_not_nest)
        key_builder.rec(key_hash, self.must_nest_graph)

    def __str__(self):
        return "LoopNestConstraints(\n" \
            "    must_nest = " + str(self.must_nest) + "\n" \
            "    must_not_nest = " + str(self.must_not_nest) + "\n" \
            "    must_nest_graph = " + str(self.must_nest_graph) + "\n" \
            ")"

# }}}

# }}}


# {{{ Initial loop nest constraint creation

# {{{ process_loop_nest_specification

def process_loop_nest_specification(
        nesting,
        max_tuple_size=None,
        complement_sets_allowed=True,
        ):

    # Ensure that user-supplied nesting conforms to syntax rules, and
    # convert string representations of nestings to tuple of UnexpandedInameSets

    import re

    def _raise_loop_nest_input_error(msg):
        valid_prio_rules = (
            "Valid `must_nest` description formats: "  # noqa
            "\"iname, iname, ...\" or (str, str, str, ...), "  # noqa
            "where str can be of form "  # noqa
            "\"iname\" or \"{iname, iname, ...}\". "  # noqa
            "No set complements allowed.\n"  # noqa
            "Valid `must_not_nest` description tuples must have length 2: "  # noqa
            "\"iname, iname\", \"iname, ~iname\", or "  # noqa
            "(str, str), where str can be of form "  # noqa
            "\"iname\", \"~iname\", \"{iname, iname, ...}\", or "  # noqa
            "\"~{iname, iname, ...}\"."  # noqa
            )
        raise ValueError(
                "Invalid loop nest prioritization: %s\n"
                "Loop nest prioritization formatting rules:\n%s"
                % (msg, valid_prio_rules))

    def _error_on_regex_match(match_str, target_str):
        if re.findall(match_str, target_str):
            _raise_loop_nest_input_error(
                "Unrecognized character(s) %s in nest string %s"
                % (re.findall(match_str, target_str), target_str))

    def _process_iname_set_str(iname_set_str):
        # Convert something like ~{i,j} or ~i or "i,j" to an UnexpandedInameSet

        # Remove leading/trailing whitespace
        iname_set_str_stripped = iname_set_str.strip()

        if not iname_set_str_stripped:
            _raise_loop_nest_input_error(
                "Found 0 inames in string %s."
                % (iname_set_str))

        # Process complement sets
        if iname_set_str_stripped[0] == "~":
            # Make sure compelement is allowed
            if not complement_sets_allowed:
                _raise_loop_nest_input_error(
                    "Complement (~) not allowed in this loop nest string %s. "
                    "If you have a use-case where allowing a currently "
                    "disallowed set complement would be helpful, and the "
                    "desired nesting constraint cannot easily be expressed "
                    "another way, "
                    "please contact the Loo.py maintainers."
                    % (iname_set_str))

            # Remove tilde
            iname_set_str_stripped = iname_set_str_stripped[1:]
            if "~" in iname_set_str_stripped:
                _raise_loop_nest_input_error(
                    "Multiple complement symbols found in iname set string %s"
                    % (iname_set_str))

            # Make sure that braces are included if multiple inames present
            if "," in iname_set_str_stripped and not (
                    iname_set_str_stripped.startswith("{") and
                    iname_set_str_stripped.endswith("}")):
                _raise_loop_nest_input_error(
                    "Complements of sets containing multiple inames must "
                    "enclose inames in braces: %s is not valid."
                    % (iname_set_str))

            complement = True
        else:
            complement = False

        # Remove leading/trailing spaces
        iname_set_str_stripped = iname_set_str_stripped.strip(" ")

        # Make sure braces are valid and strip them
        if iname_set_str_stripped[0] == "{":
            if not iname_set_str_stripped[-1] == "}":
                _raise_loop_nest_input_error(
                    "Invalid braces: %s" % (iname_set_str))
            else:
                # Remove enclosing braces
                iname_set_str_stripped = iname_set_str_stripped[1:-1]
        # (If there are dangling braces around, they will be caught next)

        # Remove any more spaces
        iname_set_str_stripped = iname_set_str_stripped.strip()

        # Should be no remaining special characters besides comma and space
        _error_on_regex_match(r"([^,\w ])", iname_set_str_stripped)

        # Split by commas or spaces to get inames
        inames = re.findall(r"([\w]+)(?:[ |,]*|$)", iname_set_str_stripped)

        # Make sure iname count matches what we expect from comma count
        if len(inames) != iname_set_str_stripped.count(",") + 1:
            _raise_loop_nest_input_error(
                "Found %d inames but expected %d in string %s."
                % (len(inames), iname_set_str_stripped.count(",") + 1,
                   iname_set_str))

        if len(inames) == 0:
            _raise_loop_nest_input_error(
                "Found empty set in string %s."
                % (iname_set_str))

        # NOTE this won't catch certain cases of bad syntax, e.g., ("{h i j,,}", "k")

        return UnexpandedInameSet(
            set([s.strip() for s in iname_set_str_stripped.split(",")]),
            complement=complement)

    if isinstance(nesting, str):
        # Enforce that constraints involving iname sets be passed as tuple.
        # Iname sets defined negatively with a *single* iname are allowed here.

        # Check for any special characters besides comma, space, and tilde.
        # E.g., curly braces would indicate that an iname set was NOT
        # passed as a tuple, which is not allowed.
        _error_on_regex_match(r"([^,\w~ ])", nesting)

        # Split by comma and process each tier
        nesting_as_tuple = tuple(
            _process_iname_set_str(set_str) for set_str in nesting.split(","))
    else:
        assert isinstance(nesting, (tuple, list))
        # Process each tier
        nesting_as_tuple = tuple(
            _process_iname_set_str(set_str) for set_str in nesting)

    # Check max_tuple_size
    if max_tuple_size and len(nesting_as_tuple) > max_tuple_size:
        _raise_loop_nest_input_error(
            "Loop nest prioritization tuple %s exceeds max tuple size %d."
            % (nesting_as_tuple))

    # Make sure nesting has len > 1
    if len(nesting_as_tuple) <= 1:
        _raise_loop_nest_input_error(
            "Loop nest prioritization tuple %s must have length > 1."
            % (nesting_as_tuple))

    # Return tuple of UnexpandedInameSets
    return nesting_as_tuple

# }}}


# {{{ constrain_loop_nesting

@for_each_kernel
def constrain_loop_nesting(
        kernel, must_nest=None, must_not_nest=None):
    r"""Add the provided constraints to the kernel.

    :arg must_nest: A tuple or comma-separated string representing
        an ordering of loop nesting tiers that must appear in the
        linearized kernel. Each item in the tuple represents a
        :class:`UnexpandedInameSet`\ s.

    :arg must_not_nest: A two-tuple or comma-separated string representing
        an ordering of loop nesting tiers that must not appear in the
        linearized kernel. Each item in the tuple represents a
        :class:`UnexpandedInameSet`\ s.
    """

    # {{{ Get any current constraints, if they exist
    if kernel.loop_nest_constraints:
        if kernel.loop_nest_constraints.must_nest:
            must_nest_constraints_old = kernel.loop_nest_constraints.must_nest
        else:
            must_nest_constraints_old = set()

        if kernel.loop_nest_constraints.must_not_nest:
            must_not_nest_constraints_old = \
                kernel.loop_nest_constraints.must_not_nest
        else:
            must_not_nest_constraints_old = set()

        if kernel.loop_nest_constraints.must_nest_graph:
            must_nest_graph_old = kernel.loop_nest_constraints.must_nest_graph
        else:
            must_nest_graph_old = {}
    else:
        must_nest_constraints_old = set()
        must_not_nest_constraints_old = set()
        must_nest_graph_old = {}

    # }}}

    # {{{ Process must_nest

    if must_nest:
        # {{{ Parse must_nest, check for conflicts, combine with old constraints

        # {{{ Parse must_nest (no complements allowed)
        must_nest_tuple = process_loop_nest_specification(
            must_nest, complement_sets_allowed=False)
        # }}}

        # {{{ Error if someone prioritizes concurrent iname

        from loopy.kernel.data import ConcurrentTag
        for iname_set in must_nest_tuple:
            for iname in iname_set.inames:
                if kernel.iname_tags_of_type(iname, ConcurrentTag):
                    raise ValueError(
                        "iname %s tagged with ConcurrentTag, "
                        "cannot use iname in must-nest constraint %s."
                        % (iname, must_nest_tuple))

        # }}}

        # {{{ Update must_nest graph (and check for cycles)

        must_nest_graph_new = add_to_must_nest_graph(
            must_nest_graph_old, must_nest_tuple, kernel.all_inames())

        # }}}

        # {{{ Make sure must_nest constraints don't violate must_not_nest
        # (this may not catch all problems)
        check_must_not_nest_against_must_nest_graph(
            must_not_nest_constraints_old, must_nest_graph_new)
        # }}}

        # {{{ Check for conflicts with inames tagged 'vec' (must be innermost)

        from loopy.kernel.data import VectorizeTag
        for iname in kernel.all_inames():
            if kernel.iname_tags_of_type(iname, VectorizeTag) and (
                    must_nest_graph_new.get(iname, set())):
                # Must-nest graph doesn't allow iname to be a leaf, error
                raise ValueError(
                    "Iname %s tagged as 'vec', but loop nest constraints "
                    "%s require that iname %s nest outside of inames %s. "
                    "Vectorized inames must nest innermost; cannot "
                    "impose loop nest specification."
                    % (iname, must_nest, iname,
                    must_nest_graph_new.get(iname, set())))

        # }}}

        # {{{ Add new must_nest constraints to existing must_nest constraints
        must_nest_constraints_new = must_nest_constraints_old | set(
            [must_nest_tuple, ])
        # }}}

        # }}}
    else:
        # {{{ No new must_nest constraints, just keep the old ones

        must_nest_constraints_new = must_nest_constraints_old
        must_nest_graph_new = must_nest_graph_old

        # }}}

    # }}}

    # {{{ Process must_not_nest

    if must_not_nest:
        # {{{ Parse must_not_nest, check for conflicts, combine with old constraints

        # {{{ Parse must_not_nest; complements allowed; max_tuple_size=2

        must_not_nest_tuple = process_loop_nest_specification(
            must_not_nest, max_tuple_size=2)

        # }}}

        # {{{ Make sure must_not_nest constraints don't violate must_nest

        # (cycles are allowed in must_not_nest constraints)
        import itertools
        must_pairs = []
        for iname_before, inames_after in must_nest_graph_new.items():
            must_pairs.extend(list(itertools.product([iname_before], inames_after)))

        if not check_must_not_nest(must_pairs, must_not_nest_tuple):
            raise ValueError(
                "constrain_loop_nesting: nest constraint conflict detected. "
                "must_not_nest constraints %s inconsistent with "
                "must_nest constraints %s."
                % (must_not_nest_tuple, must_nest_constraints_new))

        # }}}

        # {{{ Add new must_not_nest constraints to exisitng must_not_nest constraints
        must_not_nest_constraints_new = must_not_nest_constraints_old | set([
            must_not_nest_tuple, ])
        # }}}

        # }}}
    else:
        # {{{ No new must_not_nest constraints, just keep the old ones

        must_not_nest_constraints_new = must_not_nest_constraints_old

        # }}}

    # }}}

    nest_constraints = LoopNestConstraints(
        must_nest=must_nest_constraints_new,
        must_not_nest=must_not_nest_constraints_new,
        must_nest_graph=must_nest_graph_new,
        )

    return kernel.copy(loop_nest_constraints=nest_constraints)

# }}}


# {{{ add_to_must_nest_graph

def add_to_must_nest_graph(must_nest_graph, new_must_nest, all_inames):
    # Note: there should not be any complements in the new_must_nest tuples

    from copy import deepcopy
    new_graph = deepcopy(must_nest_graph)

    # First, each iname must be a node in the graph
    for missing_iname in all_inames - new_graph.keys():
        new_graph[missing_iname] = set()

    # Expand new_must_nest into (before, after) pairs
    must_nest_expanded = _expand_iname_sets_in_tuple(new_must_nest, all_inames)

    # Update must_nest_graph with new pairs
    for before, after in must_nest_expanded:
        new_graph[before].add(after)

    # Compute transitive closure
    from pytools.graph import compute_transitive_closure, contains_cycle
    new_graph_closure = compute_transitive_closure(new_graph)
    # Note: compute_transitive_closure now allows cycles, will not error

    # Check for inconsistent must_nest constraints by checking for cycle:
    if contains_cycle(new_graph_closure):
        raise ValueError(
            "add_to_must_nest_graph: Nest constraint cycle detected. "
            "must_nest constraints %s inconsistent with existing "
            "must_nest constraints %s."
            % (new_must_nest, must_nest_graph))

    return new_graph_closure

# }}}


# {{{ _expand_iname_sets_in_tuple

def _expand_iname_sets_in_tuple(
        iname_sets_tuple,
        iname_universe=None,
        ):

    # First convert UnexpandedInameSets to sets.
    # Note that must_nest constraints cannot be negatively defined.
    positively_defined_iname_sets = [
        iname_set.get_inames_represented(iname_universe)
        for iname_set in iname_sets_tuple]

    # Now expand all priority tuples into (before, after) pairs using
    # Cartesian product of all pairs of sets
    # (Assumes prio_sets length > 1, which is enforced elsewhere)
    import itertools
    loop_priority_pairs = set()
    for i, before_set in enumerate(positively_defined_iname_sets[:-1]):
        for after_set in positively_defined_iname_sets[i+1:]:
            loop_priority_pairs.update(
                list(itertools.product(before_set, after_set)))

    # Make sure no priority tuple contains an iname twice (cycle)
    for prio_tuple in loop_priority_pairs:
        if len(set(prio_tuple)) != len(prio_tuple):
            raise ValueError(
                "Loop nesting %s contains cycle: %s. "
                % (iname_sets_tuple, prio_tuple))

    return loop_priority_pairs

# }}}

# }}}


# {{{ Checking constraints

# {{{ check_must_nest

def check_must_nest(all_loop_nests, must_nest):
    r"""Determine whether must_nest constraint is satisfied by
    all_loop_nests

    :arg all_loop_nests: A sequence of sequences of inames, each representing
        the order of nested loops.

    :arg must_nest: A tuple of :class:`UnexpandedInameSet`\ s describing
        nestings that must appear in all_loop_nests.

    :returns: A :class:`bool` indicating whether the must nest constraints
        are satisfied by the provided loop nesting.
    """

    # In order to make sure must_nest is satisfied, we
    # need to expand all must_nest tiers

    # FIXME instead of expanding tiers into all pairs up front,
    # create these pairs one at a time so that we can stop as soon as we fail

    # Get all must-nest pairs of inames
    must_nest_expanded = _expand_iname_sets_in_tuple(must_nest)

    for before, after in must_nest_expanded:
        correct_pair_found = False
        for nesting in all_loop_nests:
            # If both before and after are found in the nesting,
            # and they're ordered correctly, this must-nest pair is satisfied
            if before in nesting and after in nesting and (
                    nesting.index(before) < nesting.index(after)):
                correct_pair_found = True
                break
        if not correct_pair_found:
            return False
    return True

# }}}


# {{{ check_must_not_nest

def check_must_not_nest(all_loop_nests, must_not_nest):
    r"""Determine whether must_not_nest constraint is satisfied by
    all_loop_nests

    :arg all_loop_nests: A sequence of sequences of inames, each representing
        the order of nested loops.

    :arg must_not_nest: A two-tuple of :class:`UnexpandedInameSet`\ s
        describing nestings that must not appear in all_loop_nests.

    :returns: A :class:`bool` indicating whether the must_not_nest constraints
        are satisfied by the provided loop nesting.
    """

    # Note that must_not_nest may only contain two tiers

    for nesting in all_loop_nests:
        # Go through each pair in all_loop_nests and check
        # whether it violates must not nest
        for i, iname_before in enumerate(nesting):
            for iname_after in nesting[i+1:]:
                if (must_not_nest[0].contains(iname_before)
                        and must_not_nest[1].contains(iname_after)):
                    # Stop as soon as we fail
                    return False
    return True

# }}}


# {{{ loop_nest_constraints_satisfied

def loop_nest_constraints_satisfied(
        all_loop_nests,
        must_nest_constraints=None,
        must_not_nest_constraints=None,
        ):
    r"""Determine whether must_nest and must_not_nest constraints are satisfied
    by all_loop_nests

    :arg all_loop_nests: A sequence of sequences of inames, each representing
        the order of nested loops.

    :arg must_nest_constraints: An iterable of tuples of
        :class:`UnexpandedInameSet`\ s, each describing nestings that must
        appear in all_loop_nests.

    :arg must_not_nest_constraints: An iterable of two-tuples of
        :class:`UnexpandedInameSet`\ s, each describing nestings that must not
        appear in all_loop_nests.

    :returns: A :class:`bool` indicating whether the constraints
        are satisfied by the provided loop nesting.
    """

    # Check must-nest constraints
    if must_nest_constraints:
        for must_nest in must_nest_constraints:
            if not check_must_nest(
                    all_loop_nests, must_nest):
                return False

    # Check must-not-nest constraints
    if must_not_nest_constraints:
        for must_not_nest in must_not_nest_constraints:
            if not check_must_not_nest(
                    all_loop_nests, must_not_nest):
                return False

    return True

# }}}


# {{{ check_must_not_nest_against_must_nest_graph

def check_must_not_nest_against_must_nest_graph(
        must_not_nest_constraints, must_nest_graph):
    r"""Ensure none of the must_not_nest constraints are violated by
    nestings represented in the must_nest_graph

    :arg must_not_nest_constraints: A set of two-tuples of
        :class:`UnexpandedInameSet`\ s describing nestings that must not appear
        in loop nestings.

    :arg must_nest_graph: A :class:`dict` mapping each iname to other inames
        that must be nested inside it.
    """

    if must_not_nest_constraints and must_nest_graph:
        import itertools
        must_pairs = []
        for iname_before, inames_after in must_nest_graph.items():
            must_pairs.extend(
                list(itertools.product([iname_before], inames_after)))
        if any(not check_must_not_nest(must_pairs, must_not_nest_tuple)
                for must_not_nest_tuple in must_not_nest_constraints):
            raise ValueError(
                "Nest constraint conflict detected. "
                "must_not_nest constraints %s inconsistent with "
                "must_nest relationships (must_nest graph: %s)."
                % (must_not_nest_constraints, must_nest_graph))

# }}}


# {{{ get_iname_nestings

def get_iname_nestings(linearization):
    """Return a list of iname tuples representing the deepest loop nestings
    in a kernel linearization.
    """
    from loopy.schedule import EnterLoop, LeaveLoop
    nestings = []
    current_tiers = []
    already_exiting_loops = False
    for lin_item in linearization:
        if isinstance(lin_item, EnterLoop):
            already_exiting_loops = False
            current_tiers.append(lin_item.iname)
        elif isinstance(lin_item, LeaveLoop):
            if not already_exiting_loops:
                nestings.append(tuple(current_tiers))
                already_exiting_loops = True
            del current_tiers[-1]
    return nestings

# }}}


# {{{ get_graph_sources

def get_graph_sources(graph):
    sources = set(graph.keys())
    for non_sources in graph.values():
        sources -= non_sources
    return sources

# }}}

# }}}


# {{{ Updating constraints during transformation

# {{{ replace_inames_in_nest_constraints

# {{{ (helper) _coalesce_duplicate_inames_in_one_nesting

def _coalesce_duplicate_inames_in_one_nesting(nesting, coalesce_candidates):
    r"""Coalesce inames that are duplicated in adjacent loop tiers.

    :arg nesting: A tuple containing at least two
        :class:`UnexpandedInameSet`\ s, neither of which may have
        ``complement`` attribute set to ``True``.

    :arg coalesce_candidates: A set of inames to be coalesced.

    :returns: A tuple containing :class:`UnexpandedInameSet`\ s with iname
        candidates that were duplicated in adjacent loop tiers coalesced.
        For example, if ``nesting=({ij, k}, {ij, h})`` and ``ij`` is in
        ``coalesce_candidates``, return ``({k}, {ij}, {h})``.
    """

    # For now, assume all UnexpandedInameSets have
    # complement=False, which works if we're only using this for
    # must_nest constraints since they cannot have complements
    for iname_set in nesting:
        assert not iname_set.complement

    # Copy and convert nesting to list so we can modify
    import copy
    coalesced_nesting = list(copy.deepcopy(nesting))

    # Repeat coalescing step until we don't find any adjacent pairs
    # containing duplicates (among coalesce_candidates)

    found_duplicates = True
    while found_duplicates:

        # Loop through each adjacent pair of iname_sets (adjacent
        # tiers) in nesting and coalesce in place (assume nesting has
        # at least 2 items)
        found_duplicates = False
        i = 0
        while i < len(coalesced_nesting)-1:

            # Get pair of adjacent nesting tiers (iname sets)
            iname_set_before = coalesced_nesting[i]
            iname_set_after = coalesced_nesting[i+1]

            # {{{ For each iname candidate, coalesce if necessary

            for iname in coalesce_candidates:

                # Four possible cases depending on where iname is found
                # and whether other inames exist in the tiers:

                if (iname_set_before.inames == set([iname, ]) and
                        iname_set_after.inames == set([iname, ])):
                    # Before/after both contain a lone iname to be coalesced,
                    # -> remove iname_set_after
                    del coalesced_nesting[i+1]
                    found_duplicates = True
                elif (iname_set_before.inames == set([iname, ]) and
                        iname in iname_set_after.inames):
                    # Before contains a lone iname to be coalesced,
                    # after contains iname along with others,
                    # -> remove iname from iname_set_after.inames
                    coalesced_nesting[i+1] = UnexpandedInameSet(
                        inames=iname_set_after.inames - set([iname, ]),
                        complement=iname_set_after.complement,
                        )
                    found_duplicates = True
                elif (iname in iname_set_before.inames and
                        iname_set_after.inames == set([iname, ])):
                    # After contains a lone iname to be coalesced,
                    # before contains iname along with others,
                    # -> remove iname from iname_set_before.inames
                    coalesced_nesting[i] = UnexpandedInameSet(
                        inames=iname_set_before.inames - set([iname, ]),
                        complement=iname_set_before.complement,
                        )
                    found_duplicates = True
                elif (iname in iname_set_before.inames and
                        iname in iname_set_after.inames):
                    # Before and after both contain iname along with others,
                    # -> remove iname from iname_set_{before,after}.inames
                    # and insert it in a new tier between them
                    coalesced_nesting[i] = UnexpandedInameSet(
                        inames=iname_set_before.inames - set([iname, ]),
                        complement=iname_set_before.complement,
                        )
                    coalesced_nesting[i+1] = UnexpandedInameSet(
                        inames=iname_set_after.inames - set([iname, ]),
                        complement=iname_set_after.complement,
                        )
                    coalesced_nesting.insert(i+1, UnexpandedInameSet(
                        inames=set([iname, ]),
                        complement=False,
                        ))
                    found_duplicates = True

                # Else, iname was not found in both sets, so do nothing

            # }}}

            # Increment i to move to next pair of tiers
            i = i + 1

    return tuple(coalesced_nesting)

# }}}


def replace_inames_in_nest_constraints(
        inames_to_replace, replacement_inames, old_constraints,
        coalesce_new_iname_duplicates=False,
        ):
    """Replace each iname in inames_to_replace with *all* inames in
    replacement_inames

    :arg inames_to_replace: A set of inames that may exist in
        ``old_constraints``, each of which is to be replaced with all inames
        in `replacement_inames`.

    :arg replacement_inames: A set of inames, all of which will replace each
        iname in ``inames_to_replace`` in ``old_constraints``.

    :arg old_constraints: An iterable of tuples containing one or more
        :class:`UnexpandedInameSet` objects.

    :arg coalesce_new_iname_duplicates: A :class:`bool` determining whether to
        coalesce replaced inames that are duplicated in adjacent loop tiers as
        a result of the replacement. For example, if must-nest constraint
        ``({i, k}, {j, h})`` exists, and then a transformation joins ``i`` and
        ``j`` into ``ij``  replacing ``i`` and ``j`` with ``ij`` would yield
        constraint ``({ij, k}, {ij, h})``, which contains a cycle. If
        coalescing is enabled, this new constraint will be
        changed to to ``({k}, {ij}, {h})``, which removes the cycle.

    """

    # {{{ Replace each iname in inames_to_replace w/ all inames in replacement_inames

    new_constraints = set()

    for old_nesting in old_constraints:

        # {{{ Perform replacement on each iname_set in this nesting

        new_nesting = []
        for iname_set in old_nesting:

            # Find inames to be replaced
            inames_found = inames_to_replace & iname_set.inames

            # Create the new set of inames with the replacements
            if inames_found:
                new_inames = iname_set.inames - inames_found
                new_inames.update(replacement_inames)
            else:
                new_inames = iname_set.inames.copy()

            new_nesting.append(
                UnexpandedInameSet(new_inames, iname_set.complement))

        # If we've removed things, new_nesting might only contain 1 item,
        # in which case it's meaningless and we should just exclude it
        if len(new_nesting) > 1:
            new_constraints.add(tuple(new_nesting))

        # }}}

    # }}}

    # {{{  Coalesce new iname duplicates

    # If coalesce_new_iname_duplicates=True,
    # coalesce replaced inames that are duplicated in adjacent loop tiers as
    # a result of the replacement. For example, if must-nest constraint
    # ``({i, k}, {j, h})`` exists, and then a transformation joins ``i`` and
    # ``j`` into ``ij``  replacing ``i`` and ``j`` with ``ij`` would yield
    # constraint ``({ij, k}, {ij, h})``, which contains a cycle. If
    # coalescing is enabled, this new constraint will be
    # changed to to ``({k}, {ij}, {h})``, which removes the cycle.

    if coalesce_new_iname_duplicates:

        # loop through new_constraints; handle each nesting independently
        coalesced_constraints = set()
        for new_nesting in new_constraints:
            coalesced_constraints.add(
                _coalesce_duplicate_inames_in_one_nesting(
                    new_nesting, replacement_inames))

        return coalesced_constraints
    else:
        return new_constraints

    # }}}

# }}}


# {{{ replace_inames_in_graph

def replace_inames_in_graph(
        inames_to_replace, replacement_inames, old_graph):
    """Replace each iname in inames_to_replace with all inames in
    replacement_inames"""
    # TODO more thorough docstring after initial code review

    # For any iname to replace is found as a key, their children will
    # need to become children of the replacement inames. Keep track of all
    # these children during initial replacement pass for graph values; then add
    # them as children to the new keys afterward.
    iname_to_replace_found_as_key = False
    union_of_inames_after_for_replaced_keys = set()

    new_graph = {}

    # {{{ Replace graph values

    for iname, inames_after in old_graph.items():

        # Create new inames_after (graph children) with replacements
        new_inames_after = inames_after.copy()
        inames_found = inames_to_replace & new_inames_after
        if inames_found:
            new_inames_after -= inames_found
            new_inames_after.update(replacement_inames)

        # If graph key is also one of the inames to be replaced, track its
        # children but don't insert it into the new graph (replacement key will
        # be added below).
        # Otherwise, add original iname as key with new children to new graph
        if iname in inames_to_replace:
            iname_to_replace_found_as_key = True
            union_of_inames_after_for_replaced_keys = \
                union_of_inames_after_for_replaced_keys | new_inames_after
        else:
            new_graph[iname] = new_inames_after

    # }}}

    # {{{ Add replacement inames as keys, and add appropriate children

    if iname_to_replace_found_as_key:
        for new_key in replacement_inames:
            new_graph[new_key] = union_of_inames_after_for_replaced_keys.copy()

    # }}}

    # Check for cycle
    from pytools.graph import contains_cycle
    if contains_cycle(new_graph):
        raise ValueError(
            "replace_inames_in_graph: Loop nesting constraint cycle detected. "
            "Cannot replace inames %s with inames %s."
            % (inames_to_replace, replacement_inames))

    return new_graph

# }}}


# {{{ replace_inames_in_all_nest_constraints

def replace_inames_in_all_nest_constraints(
        kernel, old_inames, new_inames,
        coalesce_new_iname_duplicates=False,
        ):
    """Replace each iname in old_inames with all inames in new_inames"""
    # TODO more thorough docstring after initial code review

    # Get old must_nest and must_not_nest, if they exist
    # (must_nest_graph will be rebuilt)
    if kernel.loop_nest_constraints:
        old_must_nest = kernel.loop_nest_constraints.must_nest
        old_must_not_nest = kernel.loop_nest_constraints.must_not_nest
        # (these could still be None)
    else:
        old_must_nest = None
        old_must_not_nest = None

    # {{{ Perform replacement on must-nest constraints

    if old_must_nest:
        # Compute new must nest constraints with replacements
        new_must_nest = replace_inames_in_nest_constraints(
            old_inames, new_inames, old_must_nest,
            coalesce_new_iname_duplicates=coalesce_new_iname_duplicates,
            )
    else:
        new_must_nest = None

    # }}}

    # {{{ Perform replacement on must-not-nest constraints

    if old_must_not_nest:
        # Compute new must not nest constraints with replacements
        new_must_not_nest = replace_inames_in_nest_constraints(
            old_inames, new_inames, old_must_not_nest,
            coalesce_new_iname_duplicates=False,
            # (for now, never coalesce must-not-nest constraints)
            )
    else:
        new_must_not_nest = None

    # }}}

    # {{{ Rebuild must_nest graph

    if new_must_nest:
        new_must_nest_graph = {}
        new_all_inames = (
            kernel.all_inames() - set(old_inames)) | set(new_inames)
        from pytools.graph import CycleError
        for must_nest_tuple in new_must_nest:
            try:
                new_must_nest_graph = add_to_must_nest_graph(
                    new_must_nest_graph, must_nest_tuple, new_all_inames)
            except CycleError:
                raise ValueError(
                    "Loop priority cycle detected when replacing inames %s "
                    "with inames %s. Previous must_nest constraints: %s"
                    % (old_inames, new_inames, old_must_nest))

        # Make sure none of the must_nest constraints violate must_not_nest
        # (this may not catch all problems)
        check_must_not_nest_against_must_nest_graph(
            new_must_not_nest, new_must_nest_graph)
    else:
        new_must_nest_graph = None

    # }}}

    return kernel.copy(
            loop_nest_constraints=LoopNestConstraints(
                must_nest=new_must_nest,
                must_not_nest=new_must_not_nest,
                must_nest_graph=new_must_nest_graph,
                )
            )

# }}}

# }}}

# }}}


# {{{ split/chunk inames

# {{{ backend

class _InameSplitter(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, within,
            iname_to_split, outer_iname, inner_iname, replacement_index):
        super().__init__(rule_mapping_context)

        self.within = within

        self.iname_to_split = iname_to_split
        self.outer_iname = outer_iname
        self.inner_iname = inner_iname

        self.replacement_index = replacement_index

    def map_reduction(self, expr, expn_state):
        if (self.iname_to_split in expr.inames
                and self.iname_to_split not in expn_state.arg_context
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction)):
            new_inames = list(expr.inames)
            new_inames.remove(self.iname_to_split)
            new_inames.extend([self.outer_iname, self.inner_iname])

            from loopy.symbolic import Reduction
            return Reduction(expr.operation, tuple(new_inames),
                        self.rec(expr.expr, expn_state),
                        expr.allow_simultaneous)
        else:
            return super().map_reduction(expr, expn_state)

    def map_variable(self, expr, expn_state):
        if (expr.name == self.iname_to_split
                and self.iname_to_split not in expn_state.arg_context
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction)):
            return self.replacement_index
        else:
            return super().map_variable(expr, expn_state)


def _split_iname_in_set(s, iname_to_split, inner_iname, outer_iname, fixed_length,
        fixed_length_is_inner):
    var_dict = s.get_var_dict()

    if iname_to_split not in var_dict:
        return s

    orig_dim_type, _ = var_dict[iname_to_split]
    # orig_dim_type may be set or param (the latter if the iname is
    # used as a parameter in a subdomain).

    # NB: dup_iname_to_split is not a globally valid identifier: only unique
    # wrt the set s.
    from pytools import generate_unique_names
    for dup_iname_to_split in generate_unique_names(f"dup_{iname_to_split}"):
        if dup_iname_to_split not in var_dict:
            break

    from loopy.isl_helpers import duplicate_axes
    s = duplicate_axes(s, (iname_to_split,), (dup_iname_to_split,))

    outer_var_nr = s.dim(orig_dim_type)
    inner_var_nr = s.dim(orig_dim_type)+1

    s = s.add_dims(orig_dim_type, 2)
    s = s.set_dim_name(orig_dim_type, outer_var_nr, outer_iname)
    s = s.set_dim_name(orig_dim_type, inner_var_nr, inner_iname)

    from loopy.isl_helpers import make_slab

    if fixed_length_is_inner:
        fixed_iname, var_length_iname = inner_iname, outer_iname
    else:
        fixed_iname, var_length_iname = outer_iname, inner_iname

    space = s.get_space()
    s = s & (
            make_slab(space, fixed_iname, 0, fixed_length)
            # name = fixed_iname + fixed_length*var_length_iname
            .add_constraint(isl.Constraint.eq_from_names(
                space, {
                    dup_iname_to_split: 1,
                    fixed_iname: -1,
                    var_length_iname: -fixed_length})))

    dup_iname_dim_type, dup_name_idx = space.get_var_dict()[dup_iname_to_split]
    s = s.project_out(dup_iname_dim_type, dup_name_idx, 1)

    return s


def _split_iname_backend(kernel, iname_to_split,
        fixed_length, fixed_length_is_inner,
        make_new_loop_index,
        outer_iname=None, inner_iname=None,
        outer_tag=None, inner_tag=None,
        slabs=(0, 0), do_tagged_check=True,
        within=None):
    """
    :arg within: If not None, limit the action of the transformation to
        matching contexts.  See :func:`loopy.match.parse_stack_match`
        for syntax.
    """

    from loopy.match import parse_match
    within = parse_match(within)

    # {{{ return the same kernel if no kernel matches

    if not any(within(kernel, insn) for insn in kernel.instructions):
        return kernel

    # }}}

    # Split inames do not inherit tags from their 'parent' inames.
    # FIXME: They *should* receive a tag that indicates that they descend from
    # an iname tagged in a certain way.
    from loopy.kernel.data import InameImplementationTag
    existing_tags = [tag
            for tag in kernel.iname_tags(iname_to_split)
            if isinstance(tag, InameImplementationTag)]
    from loopy.kernel.data import ForceSequentialTag, filter_iname_tags_by_type
    if (do_tagged_check and existing_tags
            and not filter_iname_tags_by_type(existing_tags, ForceSequentialTag)):
        raise LoopyError(f"cannot split already tagged iname '{iname_to_split}'")

    if iname_to_split not in kernel.all_inames():
        raise ValueError(
                f"cannot split loop for unknown variable '{iname_to_split}'")

    applied_iname_rewrites = kernel.applied_iname_rewrites[:]

    vng = kernel.get_var_name_generator()

    if outer_iname is None:
        outer_iname = vng(iname_to_split+"_outer")
    if inner_iname is None:
        inner_iname = vng(iname_to_split+"_inner")

    new_domains = [
            _split_iname_in_set(dom, iname_to_split, inner_iname, outer_iname,
                fixed_length, fixed_length_is_inner)
            for dom in kernel.domains]

    from pymbolic import var
    inner = var(inner_iname)
    outer = var(outer_iname)
    new_loop_index = make_new_loop_index(inner, outer)

    subst_map = {var(iname_to_split): new_loop_index}
    applied_iname_rewrites.append(subst_map)

    # {{{ update within_inames

    new_insns = []
    for insn in kernel.instructions:
        if iname_to_split in insn.within_inames and (
                within(kernel, insn)):
            new_within_inames = (
                    (insn.within_inames.copy()
                    - frozenset([iname_to_split]))
                    | frozenset([outer_iname, inner_iname]))
            insn = insn.copy(
                within_inames=new_within_inames)

        new_insns.append(insn)

    # }}}

    iname_slab_increments = kernel.iname_slab_increments.copy()
    iname_slab_increments[outer_iname] = slabs

    new_priorities = []
    for prio in kernel.loop_priority:
        new_prio = ()
        for prio_iname in prio:
            if prio_iname == iname_to_split:
                new_prio = new_prio + (outer_iname, inner_iname)
            else:
                new_prio = new_prio + (prio_iname,)
        new_priorities.append(new_prio)

    # {{{ Update nest constraints

    # Add {inner,outer} wherever iname_to_split is found in constraints.
    # Let remove_unused_inames handle removal of the old iname if necessary

    # update must_nest, must_not_nest, and must_nest_graph
    kernel = replace_inames_in_all_nest_constraints(
        kernel,
        set([iname_to_split, ]), [iname_to_split, inner_iname, outer_iname],
        )

    # }}}

    kernel = kernel.copy(
            domains=new_domains,
            iname_slab_increments=iname_slab_increments,
            instructions=new_insns,
            applied_iname_rewrites=applied_iname_rewrites,
            loop_priority=frozenset(new_priorities))

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, vng)
    ins = _InameSplitter(rule_mapping_context, within,
            iname_to_split, outer_iname, inner_iname, new_loop_index)

    from loopy.kernel.instruction import MultiAssignmentBase

    def check_insn_has_iname(kernel, insn, *args):
        return (not isinstance(insn, MultiAssignmentBase)
                or iname_to_split in insn.dependency_names()
                or iname_to_split in insn.reduction_inames())

    kernel = ins.map_kernel(kernel, within=check_insn_has_iname,
                            map_tvs=False, map_args=False)
    kernel = rule_mapping_context.finish_kernel(kernel)

    for existing_tag in existing_tags:
        kernel = tag_inames(kernel,
                {outer_iname: existing_tag, inner_iname: existing_tag})

    kernel = tag_inames(kernel, {outer_iname: outer_tag, inner_iname: inner_tag})
    kernel = remove_unused_inames(kernel, [iname_to_split])

    return kernel

# }}}


# {{{ split iname

@for_each_kernel
def split_iname(kernel, split_iname, inner_length,
        *,
        outer_iname=None, inner_iname=None,
        outer_tag=None, inner_tag=None,
        slabs=(0, 0), do_tagged_check=True,
        within=None):
    """Split *split_iname* into two inames (an 'inner' one and an 'outer' one)
    so that ``split_iname == inner + outer*inner_length`` and *inner* is of
    constant length *inner_length*.

    :arg outer_iname: The new iname to use for the 'inner' (fixed-length)
        loop. Defaults to a name derived from ``split_iname + "_outer"``
    :arg inner_iname: The new iname to use for the 'inner' (fixed-length)
        loop. Defaults to a name derived from ``split_iname + "_inner"``
    :arg inner_length: a positive integer
    :arg slabs:
        A tuple ``(head_it_count, tail_it_count)`` indicating the
        number of leading/trailing iterations of *outer_iname*
        for which separate code should be generated.
    :arg outer_tag: The iname tag (see :ref:`iname-tags`) to apply to
        *outer_iname*.
    :arg inner_tag: The iname tag (see :ref:`iname-tags`) to apply to
        *inner_iname*.
    :arg within: a stack match as understood by
        :func:`loopy.match.parse_match`.

    Split inames do not inherit tags from their 'parent' inames.
    """
    assert isinstance(kernel, LoopKernel)

    def make_new_loop_index(inner, outer):
        return inner + outer*inner_length

    return _split_iname_backend(kernel, split_iname,
            fixed_length=inner_length, fixed_length_is_inner=True,
            make_new_loop_index=make_new_loop_index,
            outer_iname=outer_iname, inner_iname=inner_iname,
            outer_tag=outer_tag, inner_tag=inner_tag,
            slabs=slabs, do_tagged_check=do_tagged_check,
            within=within)

# }}}


# {{{ chunk iname

@for_each_kernel
def chunk_iname(kernel, split_iname, num_chunks,
        outer_iname=None, inner_iname=None,
        outer_tag=None, inner_tag=None,
        slabs=(0, 0), do_tagged_check=True,
        within=None):
    """
    Split *split_iname* into two inames (an 'inner' one and an 'outer' one)
    so that ``split_iname == inner + outer*chunk_length`` and *outer* is of
    fixed length *num_chunks*.

    :arg within: a stack match as understood by
        :func:`loopy.match.parse_stack_match`.

    Split inames do not inherit tags from their 'parent' inames.

    .. versionadded:: 2016.2
    """

    size = kernel.get_iname_bounds(split_iname).size
    k0 = isl.Aff.zero_on_domain(size.domain().space)
    chunk_ceil = size.div(k0+num_chunks).ceil()
    chunk_floor = size.div(k0+num_chunks).floor()
    chunk_diff = chunk_ceil - chunk_floor
    chunk_mod = size.mod_val(num_chunks)

    from loopy.symbolic import pw_aff_to_expr
    from pymbolic.primitives import Min

    def make_new_loop_index(inner, outer):
        # These two expressions are equivalent. Benchmarking between the
        # two was inconclusive, although one is shorter.

        if 0:
            # Triggers isl issues in check pass.
            return (
                    inner +
                    pw_aff_to_expr(chunk_floor) * outer
                    +
                    pw_aff_to_expr(chunk_diff) * Min(
                        (outer, pw_aff_to_expr(chunk_mod))))
        else:
            return (
                    inner +
                    pw_aff_to_expr(chunk_ceil) * Min(
                        (outer, pw_aff_to_expr(chunk_mod)))
                    +
                    pw_aff_to_expr(chunk_floor) * (
                        outer - Min((outer, pw_aff_to_expr(chunk_mod)))))

    # {{{ check that iname is a box iname

    # Since the linearization used in the constraint used to map the domain
    # does not match the linearization in make_new_loop_index, we can't really
    # tolerate if the iname in question has constraints that make it non-boxy,
    # since these sub-indices would end up in the wrong spot.

    for dom in kernel.domains:
        var_dict = dom.get_var_dict()
        if split_iname not in var_dict:
            continue

        dt, idx = var_dict[split_iname]
        assert dt == dim_type.set

        aff_zero = isl.Aff.zero_on_domain(dom.space)
        aff_split_iname = aff_zero.set_coefficient_val(dim_type.in_, idx, 1)
        aligned_size = isl.align_spaces(size, aff_zero)
        box_dom = (
                dom
                .eliminate(dt, idx, 1)
                & aff_zero.le_set(aff_split_iname)
                & aff_split_iname.lt_set(aligned_size)
                )

        if not (
                box_dom <= dom
                and
                dom <= box_dom):
            raise LoopyError("domain '%s' is not box-shape about iname "
                    "'%s', cannot use chunk_iname()"
                    % (dom, split_iname))

    # }}}

    return _split_iname_backend(kernel, split_iname,
            fixed_length=num_chunks, fixed_length_is_inner=False,
            make_new_loop_index=make_new_loop_index,
            outer_iname=outer_iname, inner_iname=inner_iname,
            outer_tag=outer_tag, inner_tag=inner_tag,
            slabs=slabs, do_tagged_check=do_tagged_check,
            within=within)

# }}}

# }}}


# {{{ join inames

class _InameJoiner(RuleAwareSubstitutionMapper):
    def __init__(self, rule_mapping_context, within, subst_func,
            joined_inames, new_iname):
        super().__init__(rule_mapping_context,
                subst_func, within)

        self.joined_inames = set(joined_inames)
        self.new_iname = new_iname

    def map_reduction(self, expr, expn_state):
        expr_inames = set(expr.inames)
        overlap = (self.joined_inames & expr_inames
                - set(expn_state.arg_context))
        if overlap and self.within(
                expn_state.kernel,
                expn_state.instruction,
                expn_state.stack):
            if overlap != expr_inames:
                raise LoopyError(
                        "Cannot join inames '%s' if there is a reduction "
                        "that does not use all of the inames being joined. "
                        "(Found one with just '%s'.)"
                        % (
                            ", ".join(self.joined_inames),
                            ", ".join(expr_inames)))

            new_inames = expr_inames - self.joined_inames
            new_inames.add(self.new_iname)

            from loopy.symbolic import Reduction
            return Reduction(expr.operation, tuple(new_inames),
                        self.rec(expr.expr, expn_state),
                        expr.allow_simultaneous)
        else:
            return super().map_reduction(expr, expn_state)


@for_each_kernel
def join_inames(kernel, inames, new_iname=None, tag=None, within=None):
    """In a sense, the inverse of :func:`split_iname`. Takes in inames,
    finds their bounds (all but the first have to be bounded), and combines
    them into a single loop via analogs of ``new_iname = i0 * LEN(i1) + i1``.
    The old inames are re-obtained via the appropriate division/modulo
    operations.

    :arg inames: a sequence of inames, fastest varying last
    :arg within: a stack match as understood by
        :func:`loopy.match.parse_stack_match`.
    """

    from loopy.match import parse_match
    within = parse_match(within)

    # {{{ return the same kernel if no insn matches

    if not any(within(kernel, insn) for insn in kernel.instructions):
        return kernel

    # }}}

    # now fastest varying first
    inames = inames[::-1]

    if new_iname is None:
        new_iname = kernel.get_var_name_generator()("_and_".join(inames))

    from loopy.kernel.tools import DomainChanger
    domch = DomainChanger(kernel, frozenset(inames))
    for iname in inames:
        if kernel.get_home_domain_index(iname) != domch.leaf_domain_index:
            raise LoopyError("iname '%s' is not 'at home' in the "
                    "join's leaf domain" % iname)

    new_domain = domch.domain
    new_dim_idx = new_domain.dim(dim_type.set)
    new_domain = new_domain.add_dims(dim_type.set, 1)
    new_domain = new_domain.set_dim_name(dim_type.set, new_dim_idx, new_iname)

    joint_aff = zero = isl.Aff.zero_on_domain(new_domain.space)
    subst_dict = {}
    base_divisor = 1

    from pymbolic import var

    for i, iname in enumerate(inames):
        iname_dt, iname_idx = zero.get_space().get_var_dict()[iname]
        iname_aff = zero.add_coefficient_val(iname_dt, iname_idx, 1)

        joint_aff = joint_aff + base_divisor*iname_aff

        bounds = kernel.get_iname_bounds(iname, constants_only=True)

        from loopy.isl_helpers import (
                static_max_of_pw_aff, static_value_of_pw_aff)
        from loopy.symbolic import pw_aff_to_expr

        length = int(pw_aff_to_expr(
            static_max_of_pw_aff(bounds.size, constants_only=True)))

        try:
            lower_bound_aff = static_value_of_pw_aff(
                    bounds.lower_bound_pw_aff.coalesce(),
                    constants_only=False)
        except Exception as e:
            raise type(e)("while finding lower bound of '%s': " % iname)

        my_val = var(new_iname) // base_divisor
        if i+1 < len(inames):
            my_val %= length
        my_val += pw_aff_to_expr(lower_bound_aff)
        subst_dict[iname] = my_val

        base_divisor *= length

    from loopy.isl_helpers import iname_rel_aff
    new_domain = new_domain.add_constraint(
            isl.Constraint.equality_from_aff(
                iname_rel_aff(new_domain.get_space(), new_iname, "==", joint_aff)))

    for iname in inames:
        iname_to_dim = new_domain.get_space().get_var_dict()
        iname_dt, iname_idx = iname_to_dim[iname]

        if within is None:
            new_domain = new_domain.project_out(iname_dt, iname_idx, 1)

    def subst_within_inames(fid):
        result = set()
        for iname in fid:
            if iname in inames:
                result.add(new_iname)
            else:
                result.add(iname)

        return frozenset(result)

    new_insns = [
            insn.copy(
                within_inames=subst_within_inames(insn.within_inames)) if
            within(kernel, insn) else insn for insn in kernel.instructions]

    kernel = (kernel
            .copy(
                instructions=new_insns,
                domains=domch.get_domains_with(new_domain),
                applied_iname_rewrites=kernel.applied_iname_rewrites + [subst_dict]
                ))

    # {{{ Update loop nest constraints

    if kernel.loop_nest_constraints and (
            kernel.loop_nest_constraints.must_nest or
            kernel.loop_nest_constraints.must_not_nest or
            kernel.loop_nest_constraints.must_nest_graph):

        if within != parse_match(None):
            raise NotImplementedError(
                "join_inames() does not yet handle new loop nest "
                "constraints when within is not None.")

        # When joining inames, we create *implied* loop nestings.

        # {{{ Make sure that these implied nestings don't violate existing
        # constraints.

        implied_nestings = set()
        inames_orig_order = inames[::-1]  # This was reversed above
        for i, iname_before in enumerate(inames_orig_order[:-1]):
            for iname_after in inames_orig_order[i+1:]:
                implied_nestings.add((iname_before, iname_after))

        old_must_nest = kernel.loop_nest_constraints.must_nest
        old_must_not_nest = kernel.loop_nest_constraints.must_not_nest
        # (these could still be None)

        # Check to make sure special pairs don't conflict with constraints
        if old_must_nest:
            for iname_before, iname_after in implied_nestings:
                if iname_before in kernel.loop_nest_constraints.must_nest_graph[
                        iname_after]:
                    raise ValueError(
                        "Implied nestings violate existing must-nest constraints."
                        "\nimplied nestings: %s\nmust-nest constraints: %s"
                        % (implied_nestings, old_must_nest))

        if old_must_not_nest:
            if not loop_nest_constraints_satisfied(
                    implied_nestings,
                    must_not_nest_constraints=old_must_not_nest):
                raise ValueError(
                    "Implied nestings violate existing must-not-nest constraints."
                    "\nimplied nestings: %s\nmust-not-nest constraints: %s"
                    % (implied_nestings, old_must_not_nest))

        # }}}

        # When replacing inames, coalesce newly created iname duplicates
        # (see docstring for replace_inames_in_all_nest_constraints)
        kernel = replace_inames_in_all_nest_constraints(
            kernel, set(inames), [new_iname],
            coalesce_new_iname_duplicates=True,
            )
        # (will fail if cycle is created in must-nest graph)

    # }}}

    from loopy.match import parse_stack_match
    within = parse_stack_match(within)

    from pymbolic.mapper.substitutor import make_subst_func
    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    ijoin = _InameJoiner(rule_mapping_context, within,
            make_subst_func(subst_dict),
            inames, new_iname)

    kernel = rule_mapping_context.finish_kernel(
            ijoin.map_kernel(kernel))

    if tag is not None:
        kernel = tag_inames(kernel, {new_iname: tag})

    return remove_unused_inames(kernel, inames)

# }}}


# {{{ untag inames

def untag_inames(kernel, iname_to_untag, tag_type):
    """
    Remove tags on *iname_to_untag* which matches *tag_type*.

    :arg iname_to_untag: iname as string.
    :arg tag_type: a subclass of :class:`pytools.tag.Tag`, for example a
        subclass of :class:`loopy.kernel.data.InameImplementationTag`.

    .. versionadded:: 2018.1
    """
    from loopy.kernel.data import filter_iname_tags_by_type
    tags_to_remove = filter_iname_tags_by_type(
            kernel.inames[iname_to_untag].tags, tag_type)
    new_inames = kernel.inames.copy()
    new_inames[iname_to_untag] = kernel.inames[iname_to_untag].without_tags(
            tags_to_remove, verify_existence=False)

    return kernel.copy(inames=new_inames)

# }}}


# {{{ tag inames

@for_each_kernel
def tag_inames(kernel, iname_to_tag, force=False,
        ignore_nonexistent=False):
    """Tag an iname

    :arg iname_to_tag: a list of tuples ``(iname, new_tag)``. *new_tag* is given
        as an instance of a subclass of :class:`pytools.tag.Tag`, for example a
        subclass of :class:`loopy.kernel.data.InameImplementationTag`.
        May also be iterable of which, or as a string as shown in
        :ref:`iname-tags`. May also be a dictionary for backwards
        compatibility. *iname* may also be a wildcard using ``*`` and ``?``.

    .. versionchanged:: 2016.3

        Added wildcards.

    .. versionchanged:: 2018.1

        Added iterable of tags
    """

    if isinstance(iname_to_tag, str):
        def parse_kv(s):
            colon_index = s.find(":")
            if colon_index == -1:
                raise ValueError("tag decl '%s' has no colon" % s)

            return (s[:colon_index].strip(), s[colon_index+1:].strip())

        iname_to_tag = [
                parse_kv(s) for s in iname_to_tag.split(",")
                if s.strip()]

    if not iname_to_tag:
        return kernel

    # convert dict to list of tuples
    if isinstance(iname_to_tag, dict):
        iname_to_tag = list(iname_to_tag.items())

    # flatten iterables of tags for each iname

    try:
        from collections.abc import Iterable
    except ImportError:
        from collections import Iterable  # pylint:disable=no-name-in-module

    unpack_iname_to_tag = []
    for iname, tags in iname_to_tag:
        if isinstance(tags, Iterable) and not isinstance(tags, str):
            for tag in tags:
                unpack_iname_to_tag.append((iname, tag))
        else:
            unpack_iname_to_tag.append((iname, tags))
    iname_to_tag = unpack_iname_to_tag

    from loopy.kernel.data import parse_tag as inner_parse_tag

    def parse_tag(tag):
        if isinstance(tag, str):
            if tag.startswith("like."):
                tags = kernel.iname_tags(tag[5:])
                if len(tags) == 0:
                    return None
                if len(tags) == 1:
                    return tags[0]
                else:
                    raise LoopyError("cannot use like for multiple tags (for now)")
            elif tag == "unused.g":
                return find_unused_axis_tag(kernel, "g")
            elif tag == "unused.l":
                return find_unused_axis_tag(kernel, "l")

        return inner_parse_tag(tag)

    iname_to_tag = [(iname, parse_tag(tag)) for iname, tag in iname_to_tag]

    # {{{ globbing

    all_inames = kernel.all_inames()

    from loopy.match import re_from_glob
    new_iname_to_tag = {}
    for iname, new_tag in iname_to_tag:
        if "*" in iname or "?" in iname:
            match_re = re_from_glob(iname)
            for sub_iname in all_inames:
                if match_re.match(sub_iname):
                    new_iname_to_tag[sub_iname] = new_tag

        else:
            if iname not in all_inames:
                if ignore_nonexistent:
                    continue
                else:
                    raise LoopyError("iname '%s' does not exist" % iname)

            new_iname_to_tag[iname] = new_tag

    iname_to_tag = new_iname_to_tag
    del new_iname_to_tag

    # }}}

    from loopy.kernel.data import ConcurrentTag, VectorizeTag
    knl_inames = kernel.inames.copy()
    for name, new_tag in iname_to_tag.items():
        if not new_tag:
            continue

        if name not in kernel.all_inames():
            raise ValueError("cannot tag '%s'--not known" % name)

        knl_inames[name] = knl_inames[name].tagged(new_tag)

        # {{{ Check for incompatible loop nest constraints

        if isinstance(new_tag, VectorizeTag):

            # {{{ vec_inames will be nested innermost, check whether this
            # conflicts with must-nest constraints

            must_nest_graph = (kernel.loop_nest_constraints.must_nest_graph
                if kernel.loop_nest_constraints else None)

            if must_nest_graph and must_nest_graph.get(iname, set()):
                # iname is not a leaf
                raise ValueError(
                    "Loop priorities provided specify that iname %s nest "
                    "outside of inames %s, but vectorized inames "
                    "must nest innermost. Cannot tag %s with 'vec' tag."
                    % (iname, must_nest_graph.get(iname, set()), iname))

            # }}}

        elif isinstance(new_tag, ConcurrentTag) and kernel.loop_nest_constraints:

            # {{{ Don't allow tagging of must_nest iname as concurrent

            must_nest = kernel.loop_nest_constraints.must_nest

            if must_nest:
                for nesting in must_nest:
                    for iname_set in nesting:
                        if iname in iname_set.inames:
                            raise ValueError("cannot tag '%s' as concurrent--"
                                    "iname involved in must-nest constraint %s."
                                    % (iname, nesting))

            # }}}

        # }}}

    return kernel.copy(inames=knl_inames)

# }}}


# {{{ duplicate inames

class _InameDuplicator(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context,
            old_to_new, within):
        super().__init__(rule_mapping_context)

        self.old_to_new = old_to_new
        self.old_inames_set = set(old_to_new.keys())
        self.within = within

    def map_reduction(self, expr, expn_state):
        if (set(expr.inames) & self.old_inames_set
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)):
            new_inames = tuple(
                    self.old_to_new.get(iname, iname)
                    if iname not in expn_state.arg_context
                    else iname
                    for iname in expr.inames)

            from loopy.symbolic import Reduction
            return Reduction(expr.operation, new_inames,
                        self.rec(expr.expr, expn_state),
                        expr.allow_simultaneous)
        else:
            return super().map_reduction(expr, expn_state)

    def map_variable(self, expr, expn_state):
        new_name = self.old_to_new.get(expr.name)

        if (new_name is None
                or expr.name in expn_state.arg_context
                or not self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)):
            return super().map_variable(expr, expn_state)
        else:
            from pymbolic import var
            return var(new_name)

    def map_instruction(self, kernel, insn):
        if not self.within(kernel, insn, ()):
            return insn

        new_fid = frozenset(
                self.old_to_new.get(iname, iname)
                for iname in insn.within_inames)
        return insn.copy(within_inames=new_fid)


@for_each_kernel
def duplicate_inames(kernel, inames, within, new_inames=None, suffix=None,
        tags=None):
    """
    :arg within: a stack match as understood by
        :func:`loopy.match.parse_stack_match`.
    """
    if tags is None:
        tags = {}

    # {{{ normalize arguments, find unique new_inames

    if isinstance(inames, str):
        inames = [iname.strip() for iname in inames.split(",")]

    if isinstance(new_inames, str):
        new_inames = [iname.strip() for iname in new_inames.split(",")]

    from loopy.match import parse_stack_match
    within = parse_stack_match(within)

    if new_inames is None:
        new_inames = [None] * len(inames)

    if len(new_inames) != len(inames):
        raise ValueError("new_inames must have the same number of entries as inames")

    name_gen = kernel.get_var_name_generator()

    for i, iname in enumerate(inames):
        new_iname = new_inames[i]

        if new_iname is None:
            new_iname = iname

            if suffix is not None:
                new_iname += suffix

            new_iname = name_gen(new_iname)

        else:
            if name_gen.is_name_conflicting(new_iname):
                raise ValueError("new iname '%s' conflicts with existing names"
                        % new_iname)

            name_gen.add_name(new_iname)

        new_inames[i] = new_iname

    # }}}

    # {{{ duplicate the inames

    for old_iname, new_iname in zip(inames, new_inames):
        from loopy.kernel.tools import DomainChanger
        domch = DomainChanger(kernel, frozenset([old_iname]))

        # {{{ Update nest constraints

        # (don't remove any unused inames yet, that happens later)
        kernel = replace_inames_in_all_nest_constraints(
            kernel, set([old_iname, ]), [old_iname, new_iname])

        # }}}

        from loopy.isl_helpers import duplicate_axes
        kernel = kernel.copy(
                domains=domch.get_domains_with(
                    duplicate_axes(domch.domain, [old_iname], [new_iname])))

    # }}}

    # {{{ change the inames in the code

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, name_gen)
    indup = _InameDuplicator(rule_mapping_context,
            old_to_new=dict(list(zip(inames, new_inames))),
            within=within)

    kernel = rule_mapping_context.finish_kernel(
            indup.map_kernel(kernel, within=within))

    # }}}

    # {{{ realize tags

    for old_iname, new_iname in zip(inames, new_inames):
        new_tag = tags.get(old_iname)
        if new_tag is not None:
            kernel = tag_inames(kernel, {new_iname: new_tag})

    # }}}

    # Why isn't remove_unused_inames called on kernel here?

    # {{{ Remove any newly unused inames from nest constraints

    now_unused_inames = (set(inames) - get_used_inames(kernel)) & set(inames)
    kernel = replace_inames_in_all_nest_constraints(
        kernel, old_inames=now_unused_inames, new_inames=[],
        coalesce_new_iname_duplicates=False,
        )

    # }}}

    return kernel

# }}}


# {{{ iname duplication for schedulability

def _get_iname_duplication_options(insn_iname_sets, old_common_inames=frozenset([])):
    # Remove common inames of the current insn_iname_sets, as they are not relevant
    # for splitting.
    common = frozenset([]).union(*insn_iname_sets).intersection(*insn_iname_sets)

    # If common inames were found, we reduce the problem and go into recursion
    if common:
        # Remove the common inames from the instruction dependencies
        insn_iname_sets = (
            frozenset(iname_set - common for iname_set in insn_iname_sets)
            -
            frozenset([frozenset([])]))
        # Join the common inames with those previously found
        common = common.union(old_common_inames)

        # Go into recursion
        yield from _get_iname_duplication_options(insn_iname_sets, common)
        # Do not yield anything beyond here!
        return

    # Try finding a partitioning of the remaining inames, such that all instructions
    # use only inames from one of the disjoint sets from the partitioning.
    def join_sets_if_not_disjoint(sets):
        for s1 in sets:
            for s2 in sets:
                if s1 != s2 and s1 & s2:
                    return (
                        (sets - frozenset([s1, s2]))
                        | frozenset([s1 | s2])
                        ), False

        return sets, True

    partitioning = insn_iname_sets
    stop = False
    while not stop:
        partitioning, stop = join_sets_if_not_disjoint(partitioning)

    # If a partitioning was found we recursively apply this algorithm to the
    # subproblems
    if len(partitioning) > 1:
        for part in partitioning:
            working_set = frozenset(s for s in insn_iname_sets if s <= part)
            yield from _get_iname_duplication_options(working_set,
                                                         old_common_inames)
    # If exactly one set was found, an iname duplication is necessary
    elif len(partitioning) == 1:
        inames, = partitioning

        # There are splitting options for all inames
        for iname in inames:
            iname_insns = frozenset(
                    insn
                    for insn in insn_iname_sets
                    if frozenset([iname]) <= insn)

            import itertools as it
            # For a given iname, the set of instructions containing this iname
            # is inspected.  For each element of the power set without the
            # empty and the full set, one duplication option is generated.
            for insns_to_dup in it.chain.from_iterable(
                    it.combinations(iname_insns, i)
                    for i in range(1, len(iname_insns))):
                yield (
                    iname,
                    tuple(insn | old_common_inames for insn in insns_to_dup))

    # If partitioning was empty, we have recursed successfully and yield nothing


def get_iname_duplication_options(kernel, use_boostable_into=None):
    """List options for duplication of inames, if necessary for schedulability

    :returns: a generator listing all options to duplicate inames, if duplication
        of an iname is necessary to ensure the schedulability of the kernel.
        Duplication options are returned as tuples (iname, within) as
        understood by :func:`duplicate_inames`. There is no guarantee, that the
        transformed kernel will be schedulable, because multiple duplications
        of iname may be necessary.

    Some kernels require the duplication of inames in order to be schedulable, as the
    forced iname dependencies define an over-determined problem to the scheduler.
    Consider the following minimal example::

        knl = lp.make_kernel(["{[i,j]:0<=i,j<n}"],
                             \"\"\"
                             mat1[i,j] = mat1[i,j] + 1 {inames=i:j, id=i1}
                             mat2[j] = mat2[j] + 1 {inames=j, id=i2}
                             mat3[i] = mat3[i] + 1 {inames=i, id=i3}
                             \"\"\")

    In the example, there are four possibilities to resolve the problem:
    * duplicating i in instruction i3
    * duplicating i in instruction i1 and i3
    * duplicating j in instruction i2
    * duplicating i in instruction i2 and i3

    Use :func:`has_schedulable_iname_nesting` to decide whether an iname needs to be
    duplicated in a given kernel.
    """
    if isinstance(kernel, TranslationUnit):
        if len([clbl for clbl in kernel.callables_table.values() if
                isinstance(clbl, CallableKernel)]) == 1:
            kernel = kernel[list(kernel.entrypoints)[0]]

    assert isinstance(kernel, LoopKernel)

    if use_boostable_into:
        raise LoopyError("'use_boostable_into=True' is no longer supported.")

    if use_boostable_into is False:
        from warnings import warn
        warn("passing 'use_boostable_into=False' to 'get_iname_duplication_options'"
                " is deprecated. The argument will go away in 2021.",
                DeprecationWarning, stacklevel=2)

    from loopy.kernel.data import ConcurrentTag

    concurrent_inames = {
            iname
            for iname in kernel.all_inames()
            if kernel.iname_tags_of_type(iname, ConcurrentTag)}

    # First we extract the minimal necessary information from the kernel
    insn_iname_sets = (
        frozenset(
            insn.within_inames - concurrent_inames
            for insn in kernel.instructions)
        -
        frozenset([frozenset([])]))

    # Get the duplication options as a tuple of iname and a set
    for iname, insns in _get_iname_duplication_options(insn_iname_sets):
        # Check whether this iname has a parallel tag and discard it if so
        if kernel.iname_tags_of_type(iname, ConcurrentTag):
            continue

        # Reconstruct an object that may be passed to the within parameter of
        # loopy.duplicate_inames
        from loopy.match import Id, Or
        within = Or(tuple(
            Id(insn.id) for insn in kernel.instructions
            if insn.within_inames in insns))

        # Only yield the result if an instruction matched.
        if within.children:
            yield iname, within


def has_schedulable_iname_nesting(kernel):
    """
    :returns: a :class:`bool` indicating whether this kernel needs
        an iname duplication in order to be schedulable.
    """
    if isinstance(kernel, TranslationUnit):
        if len([clbl for clbl in kernel.callables_table.values() if
                isinstance(clbl, CallableKernel)]) == 1:
            kernel = kernel[list(kernel.entrypoints)[0]]
    return not bool(next(get_iname_duplication_options(kernel), False))

# }}}


# {{{ rename_inames

@for_each_kernel
def rename_iname(kernel, old_iname, new_iname, existing_ok=False, within=None):
    """
    :arg within: a stack match as understood by
        :func:`loopy.match.parse_stack_match`.
    :arg existing_ok: execute even if *new_iname* already exists
    """

    var_name_gen = kernel.get_var_name_generator()

    # FIXME: Distinguish existing iname vs. existing other variable
    does_exist = var_name_gen.is_name_conflicting(new_iname)

    if old_iname not in kernel.all_inames():
        raise LoopyError("old iname '%s' does not exist" % old_iname)

    if does_exist and not existing_ok:
        raise LoopyError("iname '%s' conflicts with an existing identifier"
                "--cannot rename" % new_iname)

    if does_exist:

        # FIXME implement loop nest constraint updating when does_exist=True
        # (otherwise constraints handled in duplicate_iname and remove_unused_inames)
        if kernel.loop_nest_constraints and (
                kernel.loop_nest_constraints.must_nest or
                kernel.loop_nest_constraints.must_not_nest or
                kernel.loop_nest_constraints.must_nest_graph):
            raise NotImplementedError(
                "rename_iname() does not yet handle new loop nest "
                "constraints when does_exist=True.")

        # {{{ check that the domains match up

        dom = kernel.get_inames_domain(frozenset((old_iname, new_iname)))

        var_dict = dom.get_var_dict()
        _, old_idx = var_dict[old_iname]
        _, new_idx = var_dict[new_iname]

        par_idx = dom.dim(dim_type.param)
        dom_old = dom.move_dims(
                dim_type.param, par_idx, dim_type.set, old_idx, 1)
        dom_old = dom_old.move_dims(
                dim_type.set, dom_old.dim(dim_type.set), dim_type.param, par_idx, 1)
        dom_old = dom_old.project_out(
                dim_type.set, new_idx if new_idx < old_idx else new_idx - 1, 1)

        par_idx = dom.dim(dim_type.param)
        dom_new = dom.move_dims(
                dim_type.param, par_idx, dim_type.set, new_idx, 1)
        dom_new = dom_new.move_dims(
                dim_type.set, dom_new.dim(dim_type.set), dim_type.param, par_idx, 1)
        dom_new = dom_new.project_out(
                dim_type.set, old_idx if old_idx < new_idx else old_idx - 1, 1)

        if not (dom_old <= dom_new and dom_new <= dom_old):
            raise LoopyError(
                    "inames {old} and {new} do not iterate over the same domain"
                    .format(old=old_iname, new=new_iname))

        # }}}

        from pymbolic import var
        subst_dict = {old_iname: var(new_iname)}

        from loopy.match import parse_stack_match
        within = parse_stack_match(within)

        from pymbolic.mapper.substitutor import make_subst_func
        rule_mapping_context = SubstitutionRuleMappingContext(
                kernel.substitutions, var_name_gen)
        smap = RuleAwareSubstitutionMapper(rule_mapping_context,
                        make_subst_func(subst_dict), within)

        kernel = rule_mapping_context.finish_kernel(
                smap.map_kernel(kernel))

        new_instructions = []
        for insn in kernel.instructions:
            if (old_iname in insn.within_inames
                    and within(kernel, insn, ())):
                insn = insn.copy(
                        within_inames=(
                            (insn.within_inames - frozenset([old_iname]))
                            | frozenset([new_iname])))

            new_instructions.append(insn)

        kernel = kernel.copy(instructions=new_instructions)

    else:
        kernel = duplicate_inames(
                kernel, [old_iname], within=within, new_inames=[new_iname])

    kernel = remove_unused_inames(kernel, [old_iname])

    return kernel

# }}}


# {{{ remove unused inames

def get_used_inames(kernel):
    import loopy as lp
    exp_kernel = lp.expand_subst(kernel)

    used_inames = set()
    for insn in exp_kernel.instructions:
        used_inames.update(
                insn.within_inames
                | insn.reduction_inames()
                | insn.sub_array_ref_inames())

    return used_inames


@for_each_kernel
def remove_unused_inames(kernel, inames=None):
    """Delete those among *inames* that are unused, i.e. project them
    out of the domain. If these inames pose implicit restrictions on
    other inames, these restrictions will persist as existentially
    quantified variables.

    :arg inames: may be an iterable of inames or a string of comma-separated inames.
    """

    # {{{ normalize arguments

    if inames is None:
        inames = kernel.all_inames()
    elif isinstance(inames, str):
        inames = inames.split(",")

    # }}}

    # {{{ check which inames are unused

    unused_inames = set(inames) - get_used_inames(kernel)

    # }}}

    # {{{ remove them

    domains = kernel.domains
    for iname in unused_inames:
        new_domains = []

        for dom in domains:
            try:
                dt, idx = dom.get_var_dict()[iname]
            except KeyError:
                pass
            else:
                dom = dom.project_out(dt, idx, 1)
            new_domains.append(dom)

        domains = new_domains

    kernel = kernel.copy(domains=domains)

    # }}}

    # {{{ Remove inames from loop nest constraints

    kernel = replace_inames_in_all_nest_constraints(
        kernel, old_inames=unused_inames, new_inames=[],
        coalesce_new_iname_duplicates=False,
        )

    # }}}

    return kernel


def remove_any_newly_unused_inames(transformation_func):
    from functools import wraps

    @wraps(transformation_func)
    def wrapper(kernel, *args, **kwargs):

        # check for remove_unused_inames argument, default: True
        remove_newly_unused_inames = kwargs.pop("remove_newly_unused_inames", True)

        if remove_newly_unused_inames:
            # call transform
            transformed_kernel = transformation_func(kernel, *args, **kwargs)

            if transformed_kernel is kernel:
                return kernel

            # determine which inames were already unused
            inames_already_unused = kernel.all_inames() - get_used_inames(kernel)

            # Remove inames that are unused due to transform
            return remove_unused_inames(
                transformed_kernel,
                transformed_kernel.all_inames()-inames_already_unused)
        else:
            # call transform
            return transformation_func(kernel, *args, **kwargs)

    return wrapper

# }}}


# {{{ split_reduction

class _ReductionSplitter(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, within, inames, direction):
        super().__init__(
                rule_mapping_context)

        self.within = within
        self.inames = inames
        self.direction = direction

    def map_reduction(self, expr, expn_state):
        if set(expr.inames) & set(expn_state.arg_context):
            # FIXME
            raise NotImplementedError()

        if (self.inames <= set(expr.inames)
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)):
            leftover_inames = set(expr.inames) - self.inames

            from loopy.symbolic import Reduction
            if self.direction == "in":
                return Reduction(expr.operation, tuple(leftover_inames),
                        Reduction(expr.operation, tuple(self.inames),
                            self.rec(expr.expr, expn_state),
                            expr.allow_simultaneous),
                        expr.allow_simultaneous)
            elif self.direction == "out":
                return Reduction(expr.operation, tuple(self.inames),
                        Reduction(expr.operation, tuple(leftover_inames),
                            self.rec(expr.expr, expn_state),
                            expr.allow_simultaneous),
                        expr.allow_simultaneous)
            else:
                raise AssertionError()
        else:
            return super().map_reduction(expr, expn_state)


def _split_reduction(kernel, inames, direction, within=None):
    if direction not in ["in", "out"]:
        raise ValueError("invalid value for 'direction': %s" % direction)

    if isinstance(inames, str):
        inames = inames.split(",")
    inames = set(inames)

    if not (inames <= kernel.all_inames()):
        raise LoopyError("Unknown inames: {}.".format(inames-kernel.all_inames()))

    from loopy.match import parse_stack_match
    within = parse_stack_match(within)

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    rsplit = _ReductionSplitter(rule_mapping_context,
            within, inames, direction)
    return rule_mapping_context.finish_kernel(
            rsplit.map_kernel(kernel))


@for_each_kernel
def split_reduction_inward(kernel, inames, within=None):
    """Takes a reduction of the form::

        sum([i,j,k], ...)

    and splits it into two nested reductions::

        sum([j,k], sum([i], ...))

    In this case, *inames* would have been ``"i"`` indicating that
    the iname ``i`` should be made the iname governing the inner reduction.

    :arg inames: A list of inames, or a comma-separated string that can
        be parsed into those
    """

    return _split_reduction(kernel, inames, "in", within)


@for_each_kernel
def split_reduction_outward(kernel, inames, within=None):
    """Takes a reduction of the form::

        sum([i,j,k], ...)

    and splits it into two nested reductions::

        sum([i], sum([j,k], ...))

    In this case, *inames* would have been ``"i"`` indicating that
    the iname ``i`` should be made the iname governing the outer reduction.

    :arg inames: A list of inames, or a comma-separated string that can
        be parsed into those
    """

    return _split_reduction(kernel, inames, "out", within)

# }}}


# {{{ affine map inames

@for_each_kernel
def affine_map_inames(kernel, old_inames, new_inames, equations):
    """Return a new *kernel* where the affine transform
    specified by *equations* has been applied to the inames.

    :arg old_inames: A list of inames to be replaced by affine transforms
        of their values.
        May also be a string of comma-separated inames.

    :arg new_inames: A list of new inames that are not yet used in *kernel*,
        but have their values established in terms of *old_inames* by
        *equations*.
        May also be a string of comma-separated inames.
    :arg equations: A list of equations estabilishing a relationship
        between *old_inames* and *new_inames*. Each equation may be
        a tuple ``(lhs, rhs)`` of expressions or a string, with left and
        right hand side of the equation separated by ``=``.
    """

    # {{{ check and parse arguments

    if isinstance(new_inames, str):
        new_inames = new_inames.split(",")
        new_inames = [iname.strip() for iname in new_inames]
    if isinstance(old_inames, str):
        old_inames = old_inames.split(",")
        old_inames = [iname.strip() for iname in old_inames]
    if isinstance(equations, str):
        equations = [equations]

    import re
    eqn_re = re.compile(r"^([^=]+)=([^=]+)$")

    def parse_equation(eqn):
        if isinstance(eqn, str):
            eqn_match = eqn_re.match(eqn)
            if not eqn_match:
                raise ValueError("invalid equation: %s" % eqn)

            from loopy.symbolic import parse
            lhs = parse(eqn_match.group(1))
            rhs = parse(eqn_match.group(2))
            return (lhs, rhs)
        elif isinstance(eqn, tuple):
            if len(eqn) != 2:
                raise ValueError("unexpected length of equation tuple, "
                        "got %d, should be 2" % len(eqn))
            return eqn
        else:
            raise ValueError("unexpected type of equation"
                    "got %d, should be string or tuple"
                    % type(eqn).__name__)

    equations = [parse_equation(eqn) for eqn in equations]

    all_vars = kernel.all_variable_names()
    for iname in new_inames:
        if iname in all_vars:
            raise LoopyError("new iname '%s' is already used in kernel"
                    % iname)

    for iname in old_inames:
        if iname not in kernel.all_inames():
            raise LoopyError("old iname '%s' not known" % iname)

    # }}}

    # {{{ substitute iname use

    from pymbolic.algorithm import solve_affine_equations_for
    old_inames_to_expr = solve_affine_equations_for(old_inames, equations)

    subst_dict = {
            v.name: expr
            for v, expr in old_inames_to_expr.items()}

    var_name_gen = kernel.get_var_name_generator()

    from pymbolic.mapper.substitutor import make_subst_func
    from loopy.match import parse_stack_match

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, var_name_gen)
    old_to_new = RuleAwareSubstitutionMapper(rule_mapping_context,
            make_subst_func(subst_dict), within=parse_stack_match(None))

    kernel = (
            rule_mapping_context.finish_kernel(
                old_to_new.map_kernel(kernel))
            .copy(
                applied_iname_rewrites=kernel.applied_iname_rewrites + [subst_dict]
                ))

    # }}}

    # {{{ change domains

    new_inames_set = frozenset(new_inames)
    old_inames_set = frozenset(old_inames)

    new_domains = []
    for idom, dom in enumerate(kernel.domains):
        dom_var_dict = dom.get_var_dict()
        old_iname_overlap = [
                iname
                for iname in old_inames
                if iname in dom_var_dict]

        if not old_iname_overlap:
            new_domains.append(dom)
            continue

        from loopy.symbolic import get_dependencies
        dom_new_inames = set()
        dom_old_inames = set()

        # mapping for new inames to dim_types
        new_iname_dim_types = {}

        dom_equations = []
        for iname in old_iname_overlap:
            for ieqn, (lhs, rhs) in enumerate(equations):
                eqn_deps = get_dependencies(lhs) | get_dependencies(rhs)
                if iname in eqn_deps:
                    dom_new_inames.update(eqn_deps & new_inames_set)
                    dom_old_inames.update(eqn_deps & old_inames_set)

                if dom_old_inames:
                    dom_equations.append((lhs, rhs))

                this_eqn_old_iname_dim_types = {
                        dom_var_dict[old_iname][0]
                        for old_iname in eqn_deps & old_inames_set}

                if this_eqn_old_iname_dim_types:
                    if len(this_eqn_old_iname_dim_types) > 1:
                        raise ValueError("inames '%s' (from equation %d (0-based)) "
                                "in domain %d (0-based) are not "
                                "of a uniform dim_type"
                                % (", ".join(eqn_deps & old_inames_set), ieqn, idom))

                    this_eqn_new_iname_dim_type, = this_eqn_old_iname_dim_types

                    for new_iname in eqn_deps & new_inames_set:
                        if new_iname in new_iname_dim_types:
                            if (this_eqn_new_iname_dim_type
                                    != new_iname_dim_types[new_iname]):
                                raise ValueError("dim_type disagreement for "
                                        "iname '%s' (from equation %d (0-based)) "
                                        "in domain %d (0-based)"
                                        % (new_iname, ieqn, idom))
                        else:
                            new_iname_dim_types[new_iname] = \
                                    this_eqn_new_iname_dim_type

        if not dom_old_inames <= set(dom_var_dict):
            raise ValueError("domain %d (0-based) does not know about "
                    "all old inames (specifically '%s') needed to define new inames"
                    % (idom, ", ".join(dom_old_inames - set(dom_var_dict))))

        # add inames to domain with correct dim_types
        dom_new_inames = list(dom_new_inames)
        for iname in dom_new_inames:
            dt = new_iname_dim_types[iname]
            iname_idx = dom.dim(dt)
            dom = dom.add_dims(dt, 1)
            dom = dom.set_dim_name(dt, iname_idx, iname)

        # add equations
        from loopy.symbolic import aff_from_expr
        for lhs, rhs in dom_equations:
            dom = dom.add_constraint(
                    isl.Constraint.equality_from_aff(
                        aff_from_expr(dom.space, rhs - lhs)))

        # project out old inames
        for iname in dom_old_inames:
            dt, idx = dom.get_var_dict()[iname]
            dom = dom.project_out(dt, idx, 1)

        new_domains.append(dom)

    # }}}

    # {{{ switch iname refs in instructions

    def fix_iname_set(insn_id, inames):
        if old_inames_set <= inames:
            return (inames - old_inames_set) | new_inames_set
        elif old_inames_set & inames:
            raise LoopyError("instruction '%s' uses only a part (%s), not all, "
                    "of the old inames"
                    % (insn_id, ", ".join(old_inames_set & inames)))
        else:
            return inames

    new_instructions = [
            insn.copy(within_inames=fix_iname_set(
                insn.id, insn.within_inames))
            for insn in kernel.instructions]

    # }}}

    return kernel.copy(domains=new_domains, instructions=new_instructions)

# }}}


# {{{ find unused axes

def find_unused_axis_tag(kernel, kind, insn_match=None):
    """For one of the hardware-parallel execution tags, find an unused
    axis.

    :arg insn_match: An instruction match as understood by
        :func:`loopy.match.parse_match`.
    :arg kind: may be "l" or "g", or the corresponding tag class name

    :returns: an :class:`loopy.kernel.data.GroupInameTag` or
        :class:`loopy.kernel.data.LocalInameTag` that is not being used within
        the instructions matched by *insn_match*.
    """
    used_axes = set()

    from loopy.kernel.data import GroupInameTag, LocalInameTag

    if isinstance(kind, str):
        found = False
        for cls in [GroupInameTag, LocalInameTag]:
            if kind == cls.print_name:
                kind = cls
                found = True
                break

        if not found:
            raise LoopyError("invlaid tag kind: %s" % kind)

    from loopy.match import parse_match
    match = parse_match(insn_match)
    insns = [insn for insn in kernel.instructions if match(kernel, insn)]

    for insn in insns:
        for iname in insn.within_inames:
            if kernel.iname_tags_of_type(iname, kind):
                used_axes.add(kind.axis)

    i = 0
    while i in used_axes:
        i += 1

    return kind(i)

# }}}


# {{{ separate_loop_head_tail_slab

# undocumented, because not super-useful
def separate_loop_head_tail_slab(kernel, iname, head_it_count, tail_it_count):
    """Mark *iname* so that the separate code is generated for
    the lower *head_it_count* and the upper *tail_it_count*
    iterations of the loop on *iname*.
    """

    iname_slab_increments = kernel.iname_slab_increments.copy()
    iname_slab_increments[iname] = (head_it_count, tail_it_count)

    return kernel.copy(iname_slab_increments=iname_slab_increments)

# }}}


# {{{ make_reduction_inames_unique

class _ReductionInameUniquifier(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, inames, within):
        super().__init__(rule_mapping_context)

        self.inames = inames
        self.old_to_new = []
        self.within = within

        self.iname_to_red_count = {}
        self.iname_to_nonsimultaneous_red_count = {}

    def map_reduction(self, expr, expn_state):
        within = self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)

        for iname in expr.inames:
            self.iname_to_red_count[iname] = (
                    self.iname_to_red_count.get(iname, 0) + 1)
            if not expr.allow_simultaneous:
                self.iname_to_nonsimultaneous_red_count[iname] = (
                    self.iname_to_nonsimultaneous_red_count.get(iname, 0) + 1)

        if within and not expr.allow_simultaneous:
            subst_dict = {}

            from pymbolic import var

            new_inames = []
            for iname in expr.inames:
                if (
                        not (self.inames is None or iname in self.inames)
                        or
                        self.iname_to_red_count[iname] <= 1):
                    new_inames.append(iname)
                    continue

                new_iname = self.rule_mapping_context.make_unique_var_name(iname)
                subst_dict[iname] = var(new_iname)
                self.old_to_new.append((iname, new_iname))
                new_inames.append(new_iname)

            from loopy.symbolic import SubstitutionMapper
            from pymbolic.mapper.substitutor import make_subst_func

            from loopy.symbolic import Reduction
            return Reduction(expr.operation, tuple(new_inames),
                    self.rec(
                        SubstitutionMapper(make_subst_func(subst_dict))(
                            expr.expr),
                        expn_state),
                    expr.allow_simultaneous)
        else:
            return super().map_reduction(
                    expr, expn_state)


@for_each_kernel
def make_reduction_inames_unique(kernel, inames=None, within=None):
    """
    :arg inames: if not *None*, only apply to these inames
    :arg within: a stack match as understood by
        :func:`loopy.match.parse_stack_match`.

    .. versionadded:: 2016.2
    """

    name_gen = kernel.get_var_name_generator()

    from loopy.match import parse_stack_match
    within = parse_stack_match(within)

    # {{{ change kernel

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, name_gen)
    r_uniq = _ReductionInameUniquifier(rule_mapping_context,
            inames, within=within)

    kernel = rule_mapping_context.finish_kernel(
            r_uniq.map_kernel(kernel))

    # }}}

    # {{{ duplicate the inames

    for old_iname, new_iname in r_uniq.old_to_new:
        from loopy.kernel.tools import DomainChanger
        domch = DomainChanger(kernel, frozenset([old_iname]))

        from loopy.isl_helpers import duplicate_axes
        kernel = kernel.copy(
                domains=domch.get_domains_with(
                    duplicate_axes(domch.domain, [old_iname], [new_iname])))

    # }}}

    return kernel

# }}}


# {{{ add_inames_to_insn

@for_each_kernel
def add_inames_to_insn(kernel, inames, insn_match):
    """
    :arg inames: a frozenset of inames that will be added to the
        instructions matched by *insn_match*, or a comma-separated
        string that parses to such a tuple.
    :arg insn_match: An instruction match as understood by
        :func:`loopy.match.parse_match`.

    :returns: an :class:`loopy.kernel.data.GroupInameTag` or
        :class:`loopy.kernel.data.LocalInameTag` that is not being used within
        the instructions matched by *insn_match*.

    .. versionadded:: 2016.3
    """

    if isinstance(inames, str):
        inames = frozenset(s.strip() for s in inames.split(","))

    if not isinstance(inames, frozenset):
        raise TypeError("'inames' must be a frozenset")

    from loopy.match import parse_match
    match = parse_match(insn_match)

    new_instructions = []

    for insn in kernel.instructions:
        if match(kernel, insn):
            new_instructions.append(
                    insn.copy(within_inames=insn.within_inames | inames))
        else:
            new_instructions.append(insn)

    return kernel.copy(instructions=new_instructions)

# }}}


@for_each_kernel
def add_inames_for_unused_hw_axes(kernel, within=None):
    """
    Returns a kernel with inames added to each instruction
    corresponding to any hardware-parallel iname tags
    (:class:`loopy.kernel.data.GroupInameTag`,
    :class:`loopy.kernel.data.LocalInameTag`) unused
    in the instruction but used elsewhere in the kernel.

    Current limitations:

    * Only one iname in the kernel may be tagged with each of the unused hw axes.
    * Occurence of an ``l.auto`` tag when an instruction is missing one of the
      local hw axes.

    :arg within: An instruction match as understood by
        :func:`loopy.match.parse_match`.
    """
    from loopy.kernel.data import (LocalInameTag, GroupInameTag,
            AutoFitLocalInameTag)

    n_local_axes = max([tag.axis
        for iname in kernel.inames.values()
        for tag in iname.tags
        if isinstance(tag, LocalInameTag)],
        default=-1) + 1

    n_group_axes = max([tag.axis
        for iname in kernel.inames.values()
        for tag in iname.tags
        if isinstance(tag, GroupInameTag)],
        default=-1) + 1

    contains_auto_local_tag = any([isinstance(tag, AutoFitLocalInameTag)
        for iname in kernel.inames.values()
        for tag in iname.tags])

    if contains_auto_local_tag:
        raise LoopyError("Kernels containing l.auto tags are invalid"
                " arguments.")

    # {{{ fill axes_to_inames

    # local_axes_to_inames: ith entry contains the iname tagged with l.i or None
    # if multiple inames are tagged with l.i
    local_axes_to_inames = []
    # group_axes_to_inames: ith entry contains the iname tagged with g.i or None
    # if multiple inames are tagged with g.i
    group_axes_to_inames = []

    for i in range(n_local_axes):
        ith_local_axes_tag = LocalInameTag(i)
        inames = [name
                for name, iname in kernel.inames.items()
                if ith_local_axes_tag in iname.tags]
        if not inames:
            raise LoopyError(f"Unused local hw axes {i}.")

        local_axes_to_inames.append(inames[0] if len(inames) == 1 else None)

    for i in range(n_group_axes):
        ith_group_axes_tag = GroupInameTag(i)
        inames = [name
                for name, iname in kernel.inames.items()
                if ith_group_axes_tag in iname.tags]
        if not inames:
            raise LoopyError(f"Unused group hw axes {i}.")

        group_axes_to_inames.append(inames[0] if len(inames) == 1 else None)

    # }}}

    from loopy.match import parse_match
    within = parse_match(within)

    new_insns = []

    for insn in kernel.instructions:
        if within(kernel, insn):
            within_tags = frozenset().union(*(kernel.inames[iname].tags
                for iname in insn.within_inames))
            missing_local_axes = [i for i in range(n_local_axes)
                    if LocalInameTag(i) not in within_tags]
            missing_group_axes = [i for i in range(n_group_axes)
                    if GroupInameTag(i) not in within_tags]

            for axis in missing_local_axes:
                iname = local_axes_to_inames[axis]
                if iname:
                    insn = insn.copy(within_inames=insn.within_inames |
                            frozenset([iname]))
                else:
                    raise LoopyError("Multiple inames tagged with l.%d while"
                            " adding unused local hw axes to instruction '%s'."
                            % (axis, insn.id))

            for axis in missing_group_axes:
                iname = group_axes_to_inames[axis]
                if iname is not None:
                    insn = insn.copy(within_inames=insn.within_inames |
                            frozenset([iname]))
                else:
                    raise LoopyError("Multiple inames tagged with g.%d while"
                            " adding unused group hw axes to instruction '%s'."
                            % (axis, insn.id))

        new_insns.append(insn)

    return kernel.copy(instructions=new_insns)

# vim: foldmethod=marker

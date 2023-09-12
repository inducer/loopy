"""Kernel object."""

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

from functools import cached_property
from enum import IntEnum
from sys import intern
from typing import (
        Dict, Sequence, Tuple, Mapping, Optional, FrozenSet, Any, Union,
        Callable, Iterator, List, Set, TYPE_CHECKING)
from dataclasses import dataclass, replace, field, fields
from warnings import warn

from collections import defaultdict

import numpy as np
from pytools import (memoize_method,
        UniqueNameGenerator, generate_unique_names, natsorted)
from pytools.tag import Taggable, Tag
import islpy as isl
from islpy import dim_type
from immutables import Map

from loopy.diagnostic import CannotBranchDomainTree, LoopyError
from loopy.tools import update_persistent_hash
from loopy.diagnostic import StaticValueFindingError
from loopy.kernel.data import (
        _ArraySeparationInfo,
        KernelArgument,
        filter_iname_tags_by_type, Iname,
        TemporaryVariable, ValueArg, ArrayArg, SubstitutionRule)
from loopy.kernel.instruction import InstructionBase
from loopy.types import LoopyType, NumpyType
from loopy.options import Options
from loopy.schedule import ScheduleItem
from loopy.typing import ExpressionT
from loopy.target import TargetBase

if TYPE_CHECKING:
    from loopy.kernel.function_interface import InKernelCallable
    from loopy.codegen import PreambleInfo


# {{{ loop kernel object

class KernelState(IntEnum):  # noqa
    INITIAL = 0
    CALLS_RESOLVED = 1
    PREPROCESSED = 2
    LINEARIZED = 3


def _get_inames_from_domains(domains):
    return frozenset().union(*
            (frozenset(dom.get_var_names(dim_type.set)) for dom in domains))


@dataclass(frozen=True)
class _BoundsRecord:
    lower_bound_pw_aff: isl.PwAff
    upper_bound_pw_aff: isl.PwAff
    size: isl.PwAff


PreambleGenerator = Callable[["PreambleInfo"], Iterator[Tuple[int, str]]]


@dataclass(frozen=True)
class LoopKernel(Taggable):
    """These correspond more or less directly to arguments of
    :func:`loopy.make_kernel`.

    .. note::

        This data structure and its attributes should be considered immutable,
        even if it contains mutable data types. See :meth:`copy` for an easy
        way of producing a modified copy.

    .. attribute:: domains

        a list of :class:`islpy.BasicSet` instances representing the
        :ref:`domain-tree`.

    .. attribute:: instructions

        A list of :class:`InstructionBase` instances, e.g.
        :class:`Assignment`. See :ref:`instructions`.

    .. attribute:: args

        A list of :class:`loopy.KernelArgument`

    .. attribute:: schedule

        *None* or a list of :class:`loopy.schedule.ScheduleItem`

    .. attribute:: name
    .. attribute:: preambles
    .. attribute:: preamble_generators
    .. attribute:: assumptions

        A :class:`islpy.BasicSet` parameter domain.

    .. attribute:: temporary_variables

        A :class:`dict` of mapping variable names to
        :class:`loopy.TemporaryVariable`
        instances.

    .. attribute:: symbol_manglers

    .. attribute:: substitutions

        a mapping from substitution names to
        :class:`SubstitutionRule` objects

    .. attribute:: iname_slab_increments

        a dictionary mapping inames to (lower_incr,
        upper_incr) tuples that will be separated out in the execution to generate
        'bulk' slabs with fewer conditionals.

    .. attribute:: loop_priority

        A frozenset of priority constraints to the kernel. Each such constraint
        is a tuple of inames. Inames occuring in such a tuple will be scheduled
        earlier than any iname following in the tuple. This applies only to inames
        with non-parallel implementation tags.

    .. attribute:: silenced_warnings

    .. attribute:: applied_iname_rewrites

        A list of past substitution dictionaries that
        were applied to the kernel. These are stored so that they may be repeated
        on expressions the user specifies later.

    .. attribute:: options

        An instance of :class:`loopy.Options`

    .. attribute:: state

        A value from :class:`KernelState`.

    .. attribute:: target

        A subclass of :class:`loopy.TargetBase`.

    .. attribute:: inames

        An instance of :class:`dict`, a mapping from the names of kernel's
        inames to their corresponding instances of :class:`loopy.kernel.data.Iname`.
        An entry is guaranteed to be present for each iname.

    .. automethod:: __call__
    .. automethod:: copy

    .. automethod:: tagged
    .. automethod:: without_tags
    """
    domains: Sequence[isl.BasicSet]
    instructions: Sequence[InstructionBase]
    args: Sequence[KernelArgument]
    assumptions: isl.BasicSet
    temporary_variables: Mapping[str, TemporaryVariable]
    inames: Mapping[str, Iname]
    substitutions: Mapping[str, SubstitutionRule]
    options: Options
    target: TargetBase
    tags: FrozenSet[Tag]
    state: KernelState = KernelState.INITIAL
    name: str = "loopy_kernel"

    preambles: Sequence[Tuple[int, str]] = ()
    preamble_generators: Sequence[PreambleGenerator] = ()
    symbol_manglers: Sequence[
            Callable[["LoopKernel", str], Optional[Tuple[LoopyType, str]]]] = ()
    linearization: Optional[Sequence[ScheduleItem]] = None
    iname_slab_increments: Mapping[str, Tuple[int, int]] = field(
            default_factory=Map)
    loop_priority: FrozenSet[Tuple[str]] = field(
            default_factory=frozenset)
    applied_iname_rewrites: Tuple[Dict[str, ExpressionT], ...] = ()
    index_dtype: NumpyType = NumpyType(np.dtype(np.int32))
    silenced_warnings: FrozenSet[str] = frozenset()

    # FIXME Yuck, this should go.
    overridden_get_grid_sizes_for_insn_ids: Optional[
            Callable[
                [FrozenSet[str],
                    Dict[str, "InKernelCallable"],
                    bool],
                Tuple[Tuple[int, ...], Tuple[int, ...]]]] = None

    def __post_init__(self):
        assert isinstance(self.assumptions, isl.BasicSet)
        assert self.assumptions.is_params()

        if not self.index_dtype.is_integral():
            raise TypeError("index_dtype must be an integer")
        if np.iinfo(self.index_dtype.numpy_dtype).min >= 0:
            raise TypeError("index_dtype must be signed")

        assert self.assumptions.get_ctx() == isl.DEFAULT_CONTEXT

    # {{{ symbol mangling

    def mangle_symbol(self, ast_builder, identifier):
        manglers = ast_builder.symbol_manglers() + list(self.symbol_manglers)

        for mangler in manglers:
            result = mangler(self, identifier)
            if result is not None:
                return result

        return None

    # }}}

    # {{{ name wrangling

    @memoize_method
    def non_iname_variable_names(self):
        return (set(self.arg_dict.keys())
                | set(self.temporary_variables.keys()))

    @memoize_method
    def all_variable_names(self):
        return (
                set(self.temporary_variables.keys())
                | set(self.substitutions.keys())
                | {arg.name for arg in self.args}
                | set(self.all_inames()))

    def get_var_name_generator(self):
        return UniqueNameGenerator(self.all_variable_names())

    def get_instruction_id_generator(self, based_on="insn"):
        used_ids = {insn.id for insn in self.instructions}

        return UniqueNameGenerator(used_ids)

    def make_unique_instruction_id(self, insns=None, based_on="insn",
            extra_used_ids=frozenset()):
        if insns is None:
            insns = self.instructions

        used_ids = {insn.id for insn in insns} | extra_used_ids

        for id_str in generate_unique_names(based_on):
            if id_str not in used_ids:
                return intern(id_str)

    def all_group_names(self):
        result = set()
        for insn in self.instructions:
            result.update(insn.groups)
            result.update(insn.conflicts_with_groups)

        return frozenset(result)

    def get_group_name_generator(self):
        return UniqueNameGenerator(set(self.all_group_names()))

    def get_var_descriptor(
            self, name: str) -> Union[TemporaryVariable, KernelArgument]:
        try:
            return self.arg_dict[name]
        except KeyError:
            pass

        try:
            return self.temporary_variables[name]
        except KeyError:
            pass

        if name in self.all_inames():
            from loopy import TemporaryVariable
            return TemporaryVariable(
                    name=name,
                    dtype=self.index_dtype,
                    shape=())

        try:
            dtype, name = self.mangle_symbol(self.target.get_device_ast_builder(),
                    name)
            return ValueArg(name, dtype)
        except TypeError:
            pass

        raise ValueError("nothing known about variable '%s'" % name)

    @cached_property
    def id_to_insn(self):
        return {insn.id: insn for insn in self.instructions}

    # }}}

    # {{{ domain wrangling

    @memoize_method
    def parents_per_domain(self) -> Sequence[Optional[int]]:
        """Return a list corresponding to self.domains (by index)
        containing domain indices which are nested around this
        domain.

        Each domains nest list walks from the leaves of the nesting
        tree to the root.
        """

        # The stack of iname sets records which inames are active
        # as we step through the linear list of domains. It also
        # determines the granularity of inames to be popped/decactivated
        # if we ascend a level.

        iname_set_stack: List[Set[str]] = []
        result: List[Optional[int]] = []

        from loopy.kernel.tools import is_domain_dependent_on_inames

        for dom_idx, dom in enumerate(self.domains):
            inames = set(dom.get_var_names(dim_type.set))

            # This next domain may be nested inside the previous domain.
            # Or it may not, in which case we need to figure out how many
            # levels of parents we need to discard in order to find the
            # true parent.

            discard_level_count = 0
            while discard_level_count < len(iname_set_stack):
                last_inames = (
                        iname_set_stack[-1-discard_level_count])
                if discard_level_count + 1 < len(iname_set_stack):
                    last_inames = (
                            last_inames - iname_set_stack[-2-discard_level_count])

                if is_domain_dependent_on_inames(self, dom_idx, last_inames):
                    break

                discard_level_count += 1

            if discard_level_count:
                iname_set_stack = iname_set_stack[:-discard_level_count]

            if result:
                parent = len(result)-1
            else:
                parent = None

            for _i in range(discard_level_count):
                assert parent is not None
                parent = result[parent]

            # found this domain's parent
            result.append(parent)

            if iname_set_stack:
                parent_inames = iname_set_stack[-1]
            else:
                parent_inames = set()
            iname_set_stack.append(parent_inames | inames)

        return result

    @memoize_method
    def all_parents_per_domain(self):
        """Return a list corresponding to self.domains (by index)
        containing domain indices which are nested around this
        domain.

        Each domains nest list walks from the leaves of the nesting
        tree to the root.
        """
        result = []

        ppd = self.parents_per_domain()
        for parent in ppd:
            # keep walking up tree to find *all* parents
            dom_result = []
            while parent is not None:
                dom_result.insert(0, parent)
                parent = ppd[parent]

            result.append(dom_result)

        return result

    @memoize_method
    def _get_home_domain_map(self) -> Mapping[str, int]:
        return {
                iname: i_domain
                for i_domain, dom in enumerate(self.domains)
                for iname in dom.get_var_names(dim_type.set)}

    def get_home_domain_index(self, iname: str) -> int:
        return self._get_home_domain_map()[iname]

    @property
    def isl_context(self) -> isl.Context:
        for dom in self.domains:
            return dom.get_ctx()

        raise AssertionError()

    @memoize_method
    def combine_domains(self, domains: Sequence[int]) -> isl.BasicSet:
        """
        :arg domains: domain indices of domains to be combined. More 'dominant'
            domains (those which get most say on the actual dim_type of an iname)
            must be later in the order.
        """
        assert isinstance(domains, tuple)  # for caching

        if not domains:
            return isl.BasicSet.universe(isl.Space.set_alloc(
                self.isl_context, 0, 0))

        result = None
        for dom_index in domains:
            dom = self.domains[dom_index]
            if result is None:
                result = dom
            else:
                aligned_dom, aligned_result = isl.align_two(
                        dom, result)
                result = aligned_result & aligned_dom

        assert result is not None
        # Subdomains may carry other domains' inames as parameters.
        # Move them back into the 'set' part of the space.
        param_names = {
                result.get_dim_name(dim_type.param, i)
                for i in range(result.dim(dim_type.param))}
        for actual_iname in param_names - self.all_params():
            result = result.move_dims(
                    dim_type.set,
                    result.dim(dim_type.set),
                    dim_type.param,
                    result.find_dim_by_name(dim_type.param, actual_iname),
                    1)

        return result

    def get_inames_domain(self, inames: FrozenSet[str]) -> isl.BasicSet:
        if not inames:
            return self.combine_domains(())

        if isinstance(inames, str):
            inames = frozenset([inames])
        if not isinstance(inames, frozenset):
            inames = frozenset(inames)

            from warnings import warn
            warn("get_inames_domain did not get a frozenset", stacklevel=2)

        return self._get_inames_domain_backend(inames)

    @memoize_method
    def get_leaf_domain_indices(self, inames):
        """Find the leaves of the domain tree needed to cover all inames.

        :arg inames: a non-mutable iterable
        """

        hdm = self._get_home_domain_map()
        ppd = self.all_parents_per_domain()

        domain_indices = set()

        # map root -> leaf
        root_to_leaf = {}

        for iname in inames:
            home_domain_index = hdm[iname]
            if home_domain_index in domain_indices:
                # nothin' new
                continue

            domain_path_to_root = [home_domain_index] + ppd[home_domain_index]
            current_root = domain_path_to_root[-1]
            previous_leaf = root_to_leaf.get(current_root)

            if previous_leaf is not None:
                # Check that we don't branch the domain tree.
                #
                # Branching the domain tree is dangerous/ill-formed because
                # it can introduce artificial restrictions on variables
                # further up the tree.

                prev_path_to_root = set([previous_leaf] + ppd[previous_leaf])
                if not prev_path_to_root <= set(domain_path_to_root):
                    raise CannotBranchDomainTree("iname set '%s' requires "
                            "branch in domain tree (when adding '%s')"
                            % (", ".join(inames), iname))
            else:
                # We're adding a new root. That's fine.
                pass

            root_to_leaf[current_root] = home_domain_index
            domain_indices.update(domain_path_to_root)

        return list(root_to_leaf.values())

    @memoize_method
    def _get_inames_domain_backend(self, inames):
        domain_indices = set()
        for leaf_dom_idx in self.get_leaf_domain_indices(inames):
            domain_indices.add(leaf_dom_idx)
            domain_indices.update(self.all_parents_per_domain()[leaf_dom_idx])

        return self.combine_domains(tuple(sorted(domain_indices)))

    # }}}

    @property
    def schedule(self):
        warn(
                "'LoopKernel.schedule' is deprecated and will be removed in 2022. "
                "Call 'LoopKernel.linearization' instead.",
                DeprecationWarning, stacklevel=2)
        return self.linearization

    # {{{ iname wrangling

    def iname_tags(self, iname):
        return self.inames[iname].tags

    def iname_tags_of_type(self, iname, tag_type_or_types,
            max_num=None, min_num=None):
        """Return a subset of *tags* that matches type *tag_type*. Raises exception
        if the number of tags found were greater than *max_num* or less than
        *min_num*.

        :arg tags: An iterable of tags.
        :arg tag_type_or_types: a subclass of :class:`loopy.kernel.data.InameTag`.
        :arg max_num: the maximum number of tags expected to be found.
        :arg min_num: the minimum number of tags expected to be found.
        """

        from loopy.kernel.data import filter_iname_tags_by_type
        return filter_iname_tags_by_type(
                self.iname_tags(iname),
                tag_type_or_types, max_num=max_num, min_num=min_num)

    @memoize_method
    def all_inames(self):
        """
        Returns a :class:`frozenset` of the names of all the inames in the kernel.
        """
        return frozenset(self.inames.keys())

    @memoize_method
    def all_params(self) -> FrozenSet[str]:
        all_inames = self.all_inames()

        result = set()
        for dom in self.domains:
            result.update(set(dom.get_var_names(dim_type.param)) - all_inames)

        from loopy.tools import intern_frozenset_of_ids
        return intern_frozenset_of_ids(result)

    def outer_params(self):
        from loopy.kernel.tools import get_outer_params
        return get_outer_params(self.domains)

    @memoize_method
    def all_insn_inames(self):
        """Return a mapping from instruction ids to inames inside which
        they should be run.
        """
        result = {}
        for insn in self.instructions:
            result[insn.id] = insn.within_inames

        return result

    @memoize_method
    def all_referenced_inames(self):
        result = set()
        for inames in self.all_insn_inames().values():
            result.update(inames)
        return result

    def insn_inames(self, insn):
        if isinstance(insn, str):
            insn = self.id_to_insn[insn]
        return insn.within_inames

    @memoize_method
    def iname_to_insns(self):
        result = {
                iname: set() for iname in self.all_inames()}
        for insn in self.instructions:
            for iname in insn.within_inames:
                result[iname].add(insn.id)

        return result

    @memoize_method
    def _remove_inames_for_shared_hw_axes(self, cond_inames):
        """
        See if cond_inames contains references to two (or more) inames that
        boil down to the same tag. If so, exclude them. (We shouldn't be writing
        conditionals for such inames because we would be implicitly restricting
        the other inames as well.)
        """

        tag_key_uses = defaultdict(list)

        from loopy.kernel.data import HardwareConcurrentTag

        for iname in cond_inames:
            tags = self.iname_tags_of_type(iname, HardwareConcurrentTag, max_num=1)
            if tags:
                tag, = tags
                tag_key_uses[tag.key].append(iname)

        multi_use_keys = {
                key for key, user_inames in tag_key_uses.items()
                if len(user_inames) > 1}

        multi_use_inames = set()
        for iname in cond_inames:
            tags = self.iname_tags_of_type(iname, HardwareConcurrentTag)
            if tags:
                tag, = filter_iname_tags_by_type(tags, HardwareConcurrentTag, 1)
                if tag.key in multi_use_keys:
                    multi_use_inames.add(iname)

        return frozenset(cond_inames - multi_use_inames)

    # }}}

    # {{{ dependency wrangling

    @memoize_method
    def recursive_insn_dep_map(self):
        """Returns a :class:`dict` mapping an instruction IDs *a*
        to all instruction IDs it directly or indirectly depends
        on.
        """

        result = {}

        def compute_deps(insn_id):
            try:
                return result[insn_id]
            except KeyError:
                pass

            insn = self.id_to_insn[insn_id]
            insn_result = set(insn.depends_on)

            for dep in list(insn.depends_on):
                insn_result.update(compute_deps(dep))

            result[insn_id] = frozenset(insn_result)
            return insn_result

        for insn in self.instructions:
            compute_deps(insn.id)

        return result

    # }}}

    # {{{ read and written variables

    @memoize_method
    def reader_map(self):
        """
        :return: a dict that maps variable names to ids of insns that read that
          variable.
        """
        result = {}

        admissible_vars = (
                {arg.name for arg in self.args}
                | set(self.temporary_variables.keys()))

        for insn in self.instructions:
            for var_name in insn.read_dependency_names() & admissible_vars:
                result.setdefault(var_name, set()).add(insn.id)

        return result

    @memoize_method
    def writer_map(self):
        """
        :return: a dict that maps variable names to ids of insns that write
            to that variable.
        """
        result = {}

        for insn in self.instructions:
            for var_name in insn.assignee_var_names():
                result.setdefault(var_name, set()).add(insn.id)

        return result

    @memoize_method
    def get_read_variables(self):
        result = set()
        for insn in self.instructions:
            result.update(insn.read_dependency_names())

        for domain in self.domains:
            result.update(domain.get_var_names(dim_type.param))

        return result

    def get_written_variables(self):
        try:
            return self._cached_written_variables
        except AttributeError:
            pass

        result = {
                var_name
                for insn in self.instructions
                for var_name in insn.assignee_var_names()}

        object.__setattr__(self, "_cached_written_variables", result)

        return result

    @memoize_method
    def get_temporary_to_base_storage_map(self):
        result = {}
        for tv in self.temporary_variables.values():
            if tv.base_storage:
                result[tv.name] = tv.base_storage

        return result

    @memoize_method
    def get_unwritten_value_args(self):
        written_vars = self.get_written_variables()

        return {
                arg.name
                for arg in self.args
                if isinstance(arg, ValueArg) and arg.name not in written_vars}

    # }}}

    # {{{ argument wrangling

    @cached_property
    def arg_dict(self) -> Dict[str, KernelArgument]:
        return {arg.name: arg for arg in self.args}

    @cached_property
    def scalar_loop_args(self):
        if self.args is None:
            return []
        else:
            from pytools import flatten
            loop_arg_names = list(flatten(dom.get_var_names(dim_type.param)
                    for dom in self.domains))
            return [arg.name for arg in self.args if isinstance(arg, ValueArg)
                    if arg.name in loop_arg_names]

    @memoize_method
    def global_var_names(self):
        from loopy.kernel.data import AddressSpace

        from loopy.kernel.data import ArrayArg
        return (
                {
                    arg.name for arg in self.args
                    if (isinstance(arg, ArrayArg)
                        and arg.address_space == AddressSpace.GLOBAL)}
                | {
                    tv.name
                    for tv in self.temporary_variables.values()
                    if tv.address_space == AddressSpace.GLOBAL})

    # }}}

    # {{{ bounds finding

    @property
    def cache_manager(self):
        try:
            return self._cache_manager
        except AttributeError:
            pass

        from loopy.kernel.tools import SetOperationCacheManager
        cm = SetOperationCacheManager()
        object.__setattr__(self, "_cache_manager", cm)
        return cm

    @memoize_method
    def get_iname_bounds(self, iname, constants_only=False):
        domain = self.get_inames_domain(frozenset([iname]))

        assumptions = self.assumptions.project_out_except(
                set(domain.get_var_dict(dim_type.param)), [dim_type.param])

        aligned_assumptions, domain = isl.align_two(assumptions, domain)

        dom_intersect_assumptions = aligned_assumptions & domain

        if constants_only:
            # Kill all variable dependencies
            dom_intersect_assumptions = dom_intersect_assumptions.project_out_except(
                    [iname], [dim_type.param, dim_type.set])

        iname_idx = dom_intersect_assumptions.get_var_dict()[iname][1]

        lower_bound_pw_aff = (
                self.cache_manager.dim_min(
                    dom_intersect_assumptions, iname_idx)
                .coalesce())
        upper_bound_pw_aff = (
                self.cache_manager.dim_max(
                    dom_intersect_assumptions, iname_idx)
                .coalesce())

        size = (upper_bound_pw_aff - lower_bound_pw_aff + 1)
        size = size.gist(assumptions)

        return _BoundsRecord(
                lower_bound_pw_aff=lower_bound_pw_aff,
                upper_bound_pw_aff=upper_bound_pw_aff,
                size=size)

    @memoize_method
    def get_constant_iname_length(self, iname):
        from loopy.isl_helpers import static_max_of_pw_aff
        from loopy.symbolic import aff_to_expr
        return int(aff_to_expr(static_max_of_pw_aff(
                self.get_iname_bounds(iname, constants_only=True).size,
                constants_only=True)))

    @memoize_method
    def get_grid_sizes_for_insn_ids_as_dicts(self, insn_ids,
            callables_table, ignore_auto=False):
        """
        Returns a tuple of (global_sizes, local_sizes), where global_sizes,
        local_sizes are the grid sizes accommodating all of *insn_ids*. The grid
        sizes are a dict from the axis index to the corresponding grid size.
        """
        all_inames_by_insns = set()
        for insn_id in insn_ids:
            all_inames_by_insns |= self.insn_inames(insn_id)

        if not all_inames_by_insns <= self.all_inames():
            raise RuntimeError("some inames collected from instructions (%s) "
                    "are not present in domain (%s)"
                    % (", ".join(sorted(all_inames_by_insns)),
                        ", ".join(sorted(self.all_inames()))))

        # {{{ include grid constraints due to callees

        global_sizes = {}
        local_sizes = {}

        from loopy.kernel.instruction import CallInstruction
        from loopy.symbolic import ResolvedFunction

        for insn_id in insn_ids:
            insn = self.id_to_insn[insn_id]
            # TODO: This might be unsafe as call-sites must be resolved to get
            # any hardware axes size constraints they might impose. However,
            # transforms like 'precompute' use this method and callables might
            # not be resolved by then.
            if (isinstance(insn, CallInstruction)
                    and isinstance(insn.expression.function, ResolvedFunction)):

                clbl = callables_table[insn.expression.function.name]
                gsize, lsize = clbl.get_hw_axes_sizes(insn.arg_id_to_arg(),
                                                      self.assumptions.space,
                                                      callables_table)

                for tgt_dict, tgt_size in [(global_sizes, gsize),
                                            (local_sizes, lsize)]:

                    for iaxis, size in tgt_size.items():
                        if iaxis in tgt_dict:
                            tgt_dict[iaxis] = tgt_dict[iaxis].max(size)
                        else:
                            tgt_dict[iaxis] = size

        # }}}

        from loopy.kernel.data import (
                GroupInameTag, LocalInameTag,
                AutoLocalInameTagBase)

        for iname in all_inames_by_insns:
            tags = self.iname_tags_of_type(
                    iname,
                    (AutoLocalInameTagBase, GroupInameTag, LocalInameTag), max_num=1)

            if not tags:
                continue

            tag, = tags

            if isinstance(tag, AutoLocalInameTagBase) and not ignore_auto:
                raise RuntimeError("cannot find grid sizes if automatic "
                        "local index tags are present")
            elif isinstance(tag, GroupInameTag):
                tgt_dict = global_sizes
            elif isinstance(tag, LocalInameTag):
                tgt_dict = local_sizes
            else:
                continue

            size = self.get_iname_bounds(iname).size

            if tag.axis in tgt_dict:
                size = tgt_dict[tag.axis].max(size)

            from loopy.isl_helpers import static_max_of_pw_aff
            try:
                # insist block size is constant
                size_as_aff = static_max_of_pw_aff(size,
                        constants_only=isinstance(tag, LocalInameTag),
                        context=self.assumptions)
                size = isl.PwAff.from_aff(size_as_aff)
            except StaticValueFindingError:
                pass

            tgt_dict[tag.axis] = size

        return global_sizes, local_sizes

    @memoize_method
    def get_grid_sizes_for_insn_ids(self, insn_ids, callables_table,
            ignore_auto=False, return_dict=False):
        """Return a tuple (global_size, local_size) containing a grid that
        could accommodate execution of all instructions whose IDs are given
        in *insn_ids*.

        :arg insn_ids: a :class:`frozenset` of instruction IDs

        *global_size* and *local_size* are :class:`islpy.PwAff` objects.
        """

        if self.overridden_get_grid_sizes_for_insn_ids:
            gsize, lsize = self.overridden_get_grid_sizes_for_insn_ids(
                insn_ids,
                callables_table=callables_table,
                ignore_auto=ignore_auto)
            if return_dict:
                return dict(enumerate(gsize)), dict(enumerate(lsize))
            else:
                return gsize, lsize

        global_sizes, local_sizes = self.get_grid_sizes_for_insn_ids_as_dicts(
                insn_ids, callables_table, ignore_auto=ignore_auto)

        if return_dict:
            return global_sizes, local_sizes

        def to_dim_tuple(size_dict, which):
            size_list = []
            sorted_axes = sorted(size_dict.keys())

            while sorted_axes:
                if sorted_axes:
                    cur_axis = sorted_axes.pop(0)
                else:
                    cur_axis = None

                assert cur_axis is not None

                if cur_axis > len(size_list):
                    raise LoopyError("%s axis %d unused for %s" % (
                        which, len(size_list), self.name))

                size_list.append(size_dict[cur_axis])

            return tuple(size_list)

        return (to_dim_tuple(global_sizes, "global"),
                to_dim_tuple(local_sizes, "local"))

    @memoize_method
    def get_grid_sizes_for_insn_ids_as_exprs(self, insn_ids,
            callables_table, ignore_auto=False, return_dict=False):
        """Return a tuple (global_size, local_size) containing a grid that
        could accommodate execution of all instructions whose IDs are given
        in *insn_ids*.

        :arg insn_ids: a :class:`frozenset` of instruction IDs

        *global_size* and *local_size* are :mod:`pymbolic` expressions
        """

        grid_size, group_size = self.get_grid_sizes_for_insn_ids(
                insn_ids, callables_table, ignore_auto, return_dict)

        if return_dict:
            def dict_to_exprs(d):
                from loopy.symbolic import pw_aff_to_expr
                return {k: pw_aff_to_expr(v, int_ok=True)
                        for k, v in d.items()}

            return dict_to_exprs(grid_size), dict_to_exprs(group_size)

        def tup_to_exprs(tup):
            from loopy.symbolic import pw_aff_to_expr
            return tuple(pw_aff_to_expr(i, int_ok=True) for i in tup)

        return tup_to_exprs(grid_size), tup_to_exprs(group_size)

    def get_grid_size_upper_bounds(self, callables_table, ignore_auto=False,
            return_dict=False):
        """Return a tuple (global_size, local_size) containing a grid that
        could accommodate execution of *all* instructions in the kernel.

        *global_size* and *local_size* are :class:`islpy.PwAff` objects.
        """
        return self.get_grid_sizes_for_insn_ids(
                frozenset(insn.id for insn in self.instructions),
                callables_table, ignore_auto=ignore_auto,
                return_dict=return_dict)

    def get_grid_size_upper_bounds_as_exprs(
            self, callables_table,
            ignore_auto=False, return_dict=False
            ) -> Tuple[Tuple[ExpressionT, ...], Tuple[ExpressionT, ...]]:
        """Return a tuple (global_size, local_size) containing a grid that
        could accommodate execution of *all* instructions in the kernel.

        *global_size* and *local_size* are :mod:`pymbolic` expressions
        """
        return self.get_grid_sizes_for_insn_ids_as_exprs(
                frozenset(insn.id for insn in self.instructions),
                callables_table, ignore_auto=ignore_auto,
                return_dict=return_dict)

    # }}}

    # {{{ local memory

    @memoize_method
    def local_var_names(self):
        from loopy.kernel.data import AddressSpace
        return {
            tv.name
            for tv in self.temporary_variables.values()
            if tv.address_space == AddressSpace.LOCAL}

    def local_mem_use(self):
        from loopy.kernel.data import AddressSpace
        return sum(
                tv.nbytes for tv in self.temporary_variables.values()
                if tv.address_space == AddressSpace.LOCAL)

    # }}}

    # {{{ nosync sets

    @memoize_method
    def get_nosync_set(self, insn_id, scope):
        assert scope in ("local", "global")

        return frozenset(
            insn_id
            for insn_id, nosync_scope in self.id_to_insn[insn_id].no_sync_with
            if nosync_scope == scope or nosync_scope == "any")

    # }}}

    # {{{ pretty-printing

    @memoize_method
    def _get_iname_order_for_printing(self):
        try:
            from loopy.kernel.tools import get_visual_iname_order_embedding
            embedding = get_visual_iname_order_embedding(self)
        except ValueError:
            from loopy.diagnostic import warn_with_kernel
            warn_with_kernel(self,
                "iname-order",
                "get_visual_iname_order_embedding() could not determine a "
                "consistent iname nesting order. This is a possible indication "
                "that the kernel may not schedule successfully, but for now "
                "it only impacts printing of the kernel.")
            embedding = {iname: iname for iname in self.all_inames()}

        return embedding

    def stringify(self, what=None, with_dependencies=False, use_separators=True,
            show_labels=True):
        all_what = {
            "name",
            "arguments",
            "domains",
            "tags",
            "variables",
            "rules",
            "instructions",
            "Dependencies",
            "linearization",
            }

        first_letter_to_what = {
                w[0]: w for w in all_what}
        assert len(first_letter_to_what) == len(all_what)

        if what is None:
            what = all_what.copy()
            if not with_dependencies:
                what.remove("Dependencies")

        if isinstance(what, str):
            if "," in what:
                what = what.split(",")
                what = {s.strip() for s in what}
            else:
                what = {
                        first_letter_to_what[w]
                        for w in what}

        if not (what <= all_what):
            raise LoopyError("invalid 'what' passed: %s"
                    % ", ".join(what-all_what))

        lines = []

        kernel = self

        if use_separators:
            sep = [75*"-"]
        else:
            sep = []

        if "name" in what:
            lines.extend(sep)
            lines.append("KERNEL: " + kernel.name)

        if "arguments" in what:
            lines.extend(sep)
            if show_labels:
                lines.append("ARGUMENTS:")
            # Arguments are ordered, do not be tempted to sort them.
            for arg in kernel.args:
                lines.append(str(arg))

        if "domains" in what:
            lines.extend(sep)
            if show_labels:
                lines.append("DOMAINS:")
            for dom, parents in zip(kernel.domains, kernel.all_parents_per_domain()):
                lines.append(len(parents)*"  " + str(dom))

        if "tags" in what:
            lines.extend(sep)
            if show_labels:
                lines.append("INAME TAGS:")
            for iname in natsorted(kernel.all_inames()):
                tags = kernel.iname_tags(iname)

                if not tags:
                    tags_str = "None"
                else:
                    tags_str = ", ".join(str(tag) for tag in tags)

                line = f"{iname}: {tags_str}"
                lines.append(line)

        if "variables" in what and kernel.temporary_variables:
            lines.extend(sep)
            if show_labels:
                lines.append("TEMPORARIES:")
            for tv in natsorted(kernel.temporary_variables.values(),
                    key=lambda key_tv: key_tv.name):
                lines.append(str(tv))

        if "rules" in what and kernel.substitutions:
            lines.extend(sep)
            if show_labels:
                lines.append("SUBSTITUTION RULES:")
            for rule_name in natsorted(kernel.substitutions.keys()):
                lines.append(str(kernel.substitutions[rule_name]))

        if "instructions" in what:
            lines.extend(sep)
            if show_labels:
                lines.append("INSTRUCTIONS:")

            from loopy.kernel.tools import stringify_instruction_list
            lines.extend(stringify_instruction_list(kernel))

        dep_lines = []
        for insn in kernel.instructions:
            if insn.depends_on:
                dep_lines.append("{} : {}".format(
                    insn.id, ",".join(insn.depends_on)))

        if "Dependencies" in what and dep_lines:
            lines.extend(sep)
            if show_labels:
                lines.append("DEPENDENCIES: "
                        "(use loopy.show_dependency_graph to visualize)")
            lines.extend(dep_lines)

        if "linearization" in what and kernel.linearization is not None:
            lines.extend(sep)
            if show_labels:
                lines.append("LINEARIZATION:")
            from loopy.schedule import dump_schedule
            lines.append(dump_schedule(kernel, kernel.linearization))

        lines.extend(sep)

        return "\n".join(lines)

    def __str__(self):
        return self.stringify()

    def __unicode__(self):
        return self.stringify()

    # }}}

    # {{{ direct execution

    def __call__(self, *args, **kwargs):
        """
        Execute the :class:`LoopKernel`.
        """
        warn("Calling a LoopKernel is deprecated, call a TranslationUnit "
                "instead.", DeprecationWarning, stacklevel=2)
        from loopy.translation_unit import make_program
        program = make_program(self)
        return program(*args, **kwargs)

    # }}}

    # {{{ pickling

    def __getstate__(self):
        result = {
                fld.name: getattr(self, fld.name)
                for fld in fields(self.__class__)
                if hasattr(self, fld.name)
                and not fld.name.startswith("_")}

        # Make the instructions lazily unpickling, to support faster
        # cache retrieval for execution.
        from loopy.kernel.instruction import _get_insn_eq_key, _get_insn_hash_key
        from loopy.tools import (
                LazilyUnpicklingListWithEqAndPersistentHashing as LazyList)

        result["instructions"] = LazyList(
                self.instructions,
                eq_key_getter=_get_insn_eq_key,
                persistent_hash_key_getter=_get_insn_hash_key)

        # Cache written variables to avoid having to unpickle instructions in
        # order to compute the written variables. This is needed on the
        # cache-to-execution path.
        result["_cached_written_variables"] = self.get_written_variables()

        # make sure that kernels are pickled with a cached hash key in place
        from loopy.tools import LoopyKeyBuilder
        LoopyKeyBuilder()(self)

        # pylint: disable=no-member
        return (result, self._pytools_persistent_hash_digest)

    def __setstate__(self, state):
        attribs, p_hash_digest = state

        for name, val in attribs.items():
            object.__setattr__(self, name, val)

        if 0:
            # {{{ check that 'reconstituted' object has same hash

            from loopy.tools import LoopyKeyBuilder
            hash_before = LoopyKeyBuilder()(self)

            object.__setattr__(
                    self, "_pytools_persistent_hash_digest", p_hash_digest)

            assert hash_before == LoopyKeyBuilder()(self)

            # }}}
        else:
            object.__setattr__(
                    self, "_pytools_persistent_hash_digest", p_hash_digest)

        from loopy.kernel.tools import SetOperationCacheManager
        object.__setattr__(self, "_cache_manager", SetOperationCacheManager())

    # }}}

    # {{{ persistent hash key generation / comparison

    hash_fields = [
            "domains",
            "instructions",
            "args",
            "assumptions",
            "temporary_variables",
            "inames",
            "substitutions",
            "options",
            "target",
            "tags",
            "state",
            "name",

            "preambles",
            # preamble_generators
            # symbol_manglers
            "linearization",
            "iname_slab_increments",
            "loop_priority",
            # applied_iname_rewrites
            "index_dtype",
            "silenced_warnings",

            # missing:
            # - applied_iname_rewrites
            #   Contains pymbolic expressions, hence a (small) headache to hash.
            #   Likely not needed for hash uniqueness => headache avoided.

            # - preamble_generators
            # - symbol_manglers
            #   These are lists of functions. It's not clear how to
            #   hash these correctly, so let's not attempt it. We'll
            #   just assume that the rest of the hash is specific enough
            #   that we won't have to rely on differences in these to
            #   resolve hash conflicts.
            ]

    update_persistent_hash = update_persistent_hash

    @memoize_method
    def __hash__(self):
        from loopy.tools import LoopyKeyBuilder
        import hashlib
        key_hash = hashlib.sha256()
        self.update_persistent_hash(key_hash, LoopyKeyBuilder())
        return hash(key_hash.digest())

    # }}}

    def get_copy_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        if "domains" in kwargs:
            inames = kwargs.get("inames", self.inames)
            domains = kwargs["domains"]
            kwargs["inames"] = {name: inames.get(name, Iname(name, frozenset()))
                                for name in _get_inames_from_domains(domains)}

            assert all(dom.get_ctx() == isl.DEFAULT_CONTEXT for dom in domains)

        return kwargs

    def copy(self, **kwargs: Any) -> "LoopKernel":
        result = replace(self, **self.get_copy_kwargs(**kwargs))

        object.__setattr__(result, "_cache_manager", self.cache_manager)

        if "instructions" not in kwargs:
            # Avoid carrying over an invalid cache when instructions are
            # modified.
            try:
                # The type system does not know about this attribute, and we're
                # not about to tell it. It's an internal caching hack.
                cwv = self._cached_written_variables  # type: ignore[attr-defined]
            except AttributeError:
                pass
            else:
                object.__setattr__(result, "_cached_written_variables", cwv)

        return result

    def _with_new_tags(self, tags) -> "LoopKernel":
        return replace(self, tags=tags)

    @memoize_method
    def _separation_info(self) -> Dict[str, _ArraySeparationInfo]:
        return {
                arg.name: arg._separation_info
                for arg in self.args
                if isinstance(arg, ArrayArg) and arg._separation_info is not None
                }

# }}}

# vim: foldmethod=marker

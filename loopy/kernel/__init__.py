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

from sys import intern

from collections import defaultdict

import numpy as np
from pytools import ImmutableRecordWithoutPickling, ImmutableRecord, memoize_method
import islpy as isl
from islpy import dim_type
import re

from pytools import UniqueNameGenerator, generate_unique_names, natsorted

from loopy.diagnostic import CannotBranchDomainTree, LoopyError
from loopy.tools import update_persistent_hash
from loopy.diagnostic import StaticValueFindingError
from loopy.kernel.data import filter_iname_tags_by_type, Iname
from warnings import warn


# {{{ unique var names

class _UniqueVarNameGenerator(UniqueNameGenerator):

    def __init__(self, existing_names=set(), forced_prefix=""):
        super().__init__(existing_names, forced_prefix)
        array_prefix_pattern = re.compile("(.*)_s[0-9]+$")

        array_prefixes = set()
        for name in existing_names:
            match = array_prefix_pattern.match(name)
            if match is None:
                continue

            array_prefixes.add(match.group(1))

        self.conflicting_array_prefixes = array_prefixes
        self.array_prefix_pattern = array_prefix_pattern

    def _name_added(self, name):
        match = self.array_prefix_pattern.match(name)
        if match is None:
            return

        self.conflicting_array_prefixes.add(match.group(1))

    def is_name_conflicting(self, name):
        if name in self.existing_names:
            return True

        # Array dimensions implemented as separate arrays generate
        # names by appending '_s<NUMBER>'. Make sure that no
        # conflicts can arise from these names.

        # Case 1: a_s0 is already a name; we are trying to insert a
        # Case 2: a is already a name; we are trying to insert a_s0

        if name in self.conflicting_array_prefixes:
            return True

        match = self.array_prefix_pattern.match(name)
        if match is None:
            return False

        return match.group(1) in self.existing_names

# }}}


# {{{ loop kernel object

class _deprecated_KernelState_SCHEDULED:  # noqa
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, klass):
        warn(
            "'KernelState.SCHEDULED' is deprecated. "
            "Use 'KernelState.LINEARIZED'.",
            DeprecationWarning, stacklevel=2)
        return self.f()

class KernelState:  # noqa
    INITIAL = 0
    CALLS_RESOLVED = 1
    PREPROCESSED = 2
    LINEARIZED = 3

    @_deprecated_KernelState_SCHEDULED
    def SCHEDULED():  # pylint:disable=no-method-argument
        return KernelState.LINEARIZED

# {{{ kernel_state, KernelState compataibility

class _deperecated_kernel_state_class_method:  # noqa
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, klass):
        warn("'temp_var_scope' is deprecated. Use 'AddressSpace'.",
                DeprecationWarning, stacklevel=2)
        return self.f()


class kernel_state:  # noqa
    """Deprecated. Use :class:`loopy.kernel.KernelState` instead.
    """

    @_deperecated_kernel_state_class_method
    def INITIAL():  # pylint:disable=no-method-argument
        return KernelState.INITIAL

    @_deperecated_kernel_state_class_method
    def PREPROCESSED():  # pylint:disable=no-method-argument
        return KernelState.PREPROCESSED

    @_deperecated_kernel_state_class_method
    def SCHEDULED():  # pylint:disable=no-method-argument
        return KernelState.SCHEDULED

# }}}


def _get_inames_from_domains(domains):
    return frozenset().union(*
            (frozenset(dom.get_var_names(dim_type.set)) for dom in domains))


class LoopKernel(ImmutableRecordWithoutPickling):
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

    .. attribute:: local_sizes
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

    .. attribute:: cache_manager
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
    """

    # {{{ constructor

    def __init__(self, domains, instructions, args=None,
            schedule=None,
            linearization=None,
            name="loopy_kernel",
            preambles=None,
            preamble_generators=None,
            assumptions=None,
            local_sizes=None,
            temporary_variables=None,
            inames=None,
            iname_to_tags=None,
            substitutions=None,
            symbol_manglers=[],

            iname_slab_increments=None,
            loop_priority=frozenset(),
            silenced_warnings=None,

            applied_iname_rewrites=None,
            cache_manager=None,
            index_dtype=None,
            options=None,

            state=KernelState.INITIAL,
            target=None,

            overridden_get_grid_sizes_for_insn_ids=None,
            _cached_written_variables=None):
        """
        :arg overridden_get_grid_sizes_for_insn_ids: A callable. When kernels get
            intersected in slab decomposition, their grid sizes shouldn't
            change. This provides a way to forward sub-kernel grid size requests.
        """

        # {{{ process constructor arguments

        if args is None:
            args = []
        if preambles is None:
            preambles = []
        if preamble_generators is None:
            preamble_generators = []
        if local_sizes is None:
            local_sizes = {}
        if temporary_variables is None:
            temporary_variables = {}
        if substitutions is None:
            substitutions = {}
        if iname_slab_increments is None:
            iname_slab_increments = {}

        if silenced_warnings is None:
            silenced_warnings = []
        if applied_iname_rewrites is None:
            applied_iname_rewrites = []

        if cache_manager is None:
            from loopy.kernel.tools import SetOperationCacheManager
            cache_manager = SetOperationCacheManager()

        if iname_to_tags is not None:
            warn("Providing iname_to_tags is deprecated, pass inames instead. "
                    "Will be unsupported in 2022.",
                    DeprecationWarning, stacklevel=2)

        if inames is None:
            if iname_to_tags is None:
                iname_to_tags = {}

            inames = {name: Iname(name, iname_to_tags.get(name, frozenset()))
                      for name in _get_inames_from_domains(domains)}
        else:
            if iname_to_tags is not None:
                raise LoopyError("Cannot provide both iname_to_tags and inames to "
                        "LoopKernel.__init__")

            inames = {
                name: inames.get(name, Iname(name, frozenset()))
                for name in _get_inames_from_domains(domains)}

        if index_dtype is None:
            index_dtype = np.int32

        # }}}

        # {{{ process assumptions

        if assumptions is None:
            dom0_space = domains[0].get_space()
            assumptions_space = isl.Space.params_alloc(
                    dom0_space.get_ctx(), dom0_space.dim(dim_type.param))
            for i in range(dom0_space.dim(dim_type.param)):
                assumptions_space = assumptions_space.set_dim_name(
                        dim_type.param, i,
                        dom0_space.get_dim_name(dim_type.param, i))
            assumptions = isl.BasicSet.universe(assumptions_space)

        elif isinstance(assumptions, str):
            assumptions_set_str = "[%s] -> { : %s}" \
                    % (",".join(s for s in self.outer_params(domains)),
                        assumptions)
            assumptions = isl.BasicSet.read_from_str(domains[0].get_ctx(),
                    assumptions_set_str)

        assert assumptions.is_params()

        # }}}

        from loopy.types import to_loopy_type
        index_dtype = to_loopy_type(index_dtype, target=target)
        if not index_dtype.is_integral():
            raise TypeError("index_dtype must be an integer")
        if np.iinfo(index_dtype.numpy_dtype).min >= 0:
            raise TypeError("index_dtype must be signed")

        if state not in [
                KernelState.INITIAL,
                KernelState.CALLS_RESOLVED,
                KernelState.PREPROCESSED,
                KernelState.LINEARIZED,
                ]:
            raise ValueError("invalid value for 'state'")

        # `linearization` is replacing `schedule`, but we're not changing
        # this under the hood yet, so for now, store it inside `schedule`
        # and raise deprecation warning anyway
        if schedule is not None:
            if linearization is not None:
                # these should not both be present
                raise ValueError(
                    "received both `schedule` and `linearization` args, "
                    "'LoopKernel.schedule' is deprecated. "
                    "Use 'LoopKernel.linearization'.")
            warn(
                "'LoopKernel.schedule' is deprecated. "
                "Use 'LoopKernel.linearization'.",
                DeprecationWarning, stacklevel=2)
        elif linearization is not None:
            schedule = linearization

        assert all(dom.get_ctx() == isl.DEFAULT_CONTEXT for dom in domains)
        assert assumptions.get_ctx() == isl.DEFAULT_CONTEXT

        ImmutableRecordWithoutPickling.__init__(self,
                domains=domains,
                instructions=instructions,
                args=args,
                schedule=schedule,
                name=name,
                preambles=preambles,
                preamble_generators=preamble_generators,
                assumptions=assumptions,
                iname_slab_increments=iname_slab_increments,
                loop_priority=loop_priority,
                silenced_warnings=silenced_warnings,
                temporary_variables=temporary_variables,
                local_sizes=local_sizes,
                inames=inames,
                substitutions=substitutions,
                cache_manager=cache_manager,
                applied_iname_rewrites=applied_iname_rewrites,
                symbol_manglers=symbol_manglers,
                index_dtype=index_dtype,
                options=options,
                state=state,
                target=target,
                overridden_get_grid_sizes_for_insn_ids=(
                    overridden_get_grid_sizes_for_insn_ids),
                _cached_written_variables=_cached_written_variables)

        self._kernel_executor_cache = {}

    # }}}

    # {{{ symbol mangling

    def mangle_symbol(self, ast_builder, identifier):
        manglers = ast_builder.symbol_manglers() + self.symbol_manglers

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
    def all_variable_names(self, include_temp_storage=True):
        return (
                set(self.temporary_variables.keys())
                | {tv.base_storage
                    for tv in self.temporary_variables.values()
                    if tv.base_storage is not None and include_temp_storage}
                | set(self.substitutions.keys())
                | {arg.name for arg in self.args}
                | set(self.all_inames()))

    def get_var_name_generator(self):
        return _UniqueVarNameGenerator(self.all_variable_names())

    def get_instruction_id_generator(self, based_on="insn"):
        used_ids = {insn.id for insn in self.instructions}

        return UniqueNameGenerator(used_ids)

    def make_unique_instruction_id(self, insns=None, based_on="insn",
            extra_used_ids=set()):
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
        return _UniqueVarNameGenerator(set(self.all_group_names()))

    def get_var_descriptor(self, name):
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
            from loopy import ValueArg
            return ValueArg(name, dtype)
        except TypeError:
            pass

        raise ValueError("nothing known about variable '%s'" % name)

    @property
    @memoize_method
    def id_to_insn(self):
        return {insn.id: insn for insn in self.instructions}

    # }}}

    # {{{ domain wrangling

    @memoize_method
    def parents_per_domain(self):
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

        iname_set_stack = []
        result = []

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

            for i in range(discard_level_count):
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
        for dom, parent in zip(self.domains, ppd):
            # keep walking up tree to find *all* parents
            dom_result = []
            while parent is not None:
                dom_result.insert(0, parent)
                parent = ppd[parent]

            result.append(dom_result)

        return result

    @memoize_method
    def _get_home_domain_map(self):
        return {
                iname: i_domain
                for i_domain, dom in enumerate(self.domains)
                for iname in dom.get_var_names(dim_type.set)}

    def get_home_domain_index(self, iname):
        return self._get_home_domain_map()[iname]

    @property
    def isl_context(self):
        for dom in self.domains:
            return dom.get_ctx()

        assert False

    @memoize_method
    def combine_domains(self, domains):
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

        return result

    def get_inames_domain(self, inames):
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

    # {{{ iname wrangling

    @property
    @memoize_method
    def iname_to_tags(self):
        warn(
                "LoopKernel.iname_to_tags is deprecated. "
                "Call LoopKernel.inames instead, "
                "will be unsupported in 2022.",
                DeprecationWarning, stacklevel=2)
        return {name: iname.tags
                for name, iname in self.inames.items()
                if iname.tags}

    def iname_tags(self, iname):
        return self.inames[iname].tags

    def iname_tags_of_type(self, iname, tag_type_or_types,
            max_num=None, min_num=None):
        """Return a subset of *tags* that matches type *tag_type*. Raises exception
        if the number of tags found were greater than *max_num* or less than
        *min_num*.

        :arg tags: An iterable of tags.
        :arg tag_type_or_types: a subclass of :class:`loopy.kernel.data.IndexTag`.
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
    def all_params(self):
        all_inames = self.all_inames()

        result = set()
        for dom in self.domains:
            result.update(set(dom.get_var_names(dim_type.param)) - all_inames)

        from loopy.tools import intern_frozenset_of_ids
        return intern_frozenset_of_ids(result)

    def outer_params(self, domains=None):
        if domains is None:
            domains = self.domains

        all_inames = set()
        all_params = set()
        for dom in domains:
            all_inames.update(dom.get_var_names(dim_type.set))
            all_params.update(dom.get_var_names(dim_type.param))

        from loopy.tools import intern_frozenset_of_ids
        return intern_frozenset_of_ids(all_params-all_inames)

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

    @memoize_method
    def get_written_variables(self):
        if self._cached_written_variables is not None:
            return self._cached_written_variables

        return frozenset(
                var_name
                for insn in self.instructions
                for var_name in insn.assignee_var_names())

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

        from loopy.kernel.data import ValueArg
        return {
                arg.name
                for arg in self.args
                if isinstance(arg, ValueArg) and arg.name not in written_vars}

    # }}}

    # {{{ argument wrangling

    @property
    @memoize_method
    def arg_dict(self):
        return {arg.name: arg for arg in self.args}

    @property
    @memoize_method
    def scalar_loop_args(self):
        from loopy.kernel.data import ValueArg

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

        class BoundsRecord(ImmutableRecord):
            pass

        size = (upper_bound_pw_aff - lower_bound_pw_aff + 1)
        size = size.gist(assumptions)

        return BoundsRecord(
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

        from loopy.kernel.data import ValueArg
        from loopy.kernel.instruction import CallInstruction
        from loopy.kernel.function_interface import (CallableKernel,
                get_kw_pos_association)
        from loopy.isl_helpers import subst_into_pwaff
        from loopy.symbolic import ResolvedFunction

        for insn in self.instructions:
            if isinstance(insn, CallInstruction) and isinstance(
                    insn.expression.function, ResolvedFunction):
                clbl = callables_table[insn.expression.function.name]
                if isinstance(clbl, CallableKernel):
                    _, pos_to_kw = get_kw_pos_association(clbl.subkernel)
                    subst_dict = {
                            pos_to_kw[i]: param
                            for i, param in enumerate(insn.expression.parameters)
                            if isinstance(clbl.subkernel.arg_dict[pos_to_kw[i]],
                                          ValueArg)}

                    gsize, lsize = (
                            clbl.subkernel.get_grid_sizes_for_insn_ids_as_dicts(
                                frozenset(insn.id
                                          for insn in clbl.subkernel.instructions),
                                callables_table, ignore_auto))

                    for tgt_dict, tgt_size in [(global_sizes, gsize),
                                                (local_sizes, lsize)]:

                        for iaxis, size in tgt_size.items():
                            size = subst_into_pwaff(self.assumptions.space,
                                    size, subst_dict)
                            if iaxis in tgt_dict:
                                tgt_dict[iaxis] = tgt_dict[iaxis].max(size)
                            else:
                                tgt_dict[iaxis] = size

        # }}}

        from loopy.kernel.data import (
                GroupIndexTag, LocalIndexTag,
                AutoLocalIndexTagBase)

        for iname in all_inames_by_insns:
            tags = self.iname_tags_of_type(
                    iname,
                    (AutoLocalIndexTagBase, GroupIndexTag, LocalIndexTag), max_num=1)

            if not tags:
                continue

            tag, = tags

            if isinstance(tag, AutoLocalIndexTagBase) and not ignore_auto:
                raise RuntimeError("cannot find grid sizes if automatic "
                        "local index tags are present")
            elif isinstance(tag, GroupIndexTag):
                tgt_dict = global_sizes
            elif isinstance(tag, LocalIndexTag):
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
                        constants_only=isinstance(tag, LocalIndexTag),
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
            return self.overridden_get_grid_sizes_for_insn_ids(
                    insn_ids,
                    callables_table=callables_table,
                    ignore_auto=ignore_auto)

        global_sizes, local_sizes = self.get_grid_sizes_for_insn_ids_as_dicts(
                insn_ids, callables_table, ignore_auto=ignore_auto)

        if return_dict:
            return global_sizes, local_sizes

        def to_dim_tuple(size_dict, which, forced_sizes={}):
            forced_sizes = forced_sizes.copy()

            size_list = []
            sorted_axes = sorted(size_dict.keys())

            while sorted_axes or forced_sizes:
                if sorted_axes:
                    cur_axis = sorted_axes.pop(0)
                else:
                    cur_axis = None

                if len(size_list) in forced_sizes:
                    size_list.append(forced_sizes.pop(len(size_list)))
                    continue

                assert cur_axis is not None

                if cur_axis > len(size_list):
                    raise LoopyError("%s axis %d unused for %s" % (
                        which, len(size_list), self.name))

                size_list.append(size_dict[cur_axis])

            return tuple(size_list)

        return (to_dim_tuple(global_sizes, "global"),
                to_dim_tuple(local_sizes, "local", forced_sizes=self.local_sizes))

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
                callables_table, ignore_auto=ignore_auto)

    def get_grid_size_upper_bounds_as_exprs(self, callables_table,
            ignore_auto=False, return_dict=False):
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
            "schedule",
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
            for arg_name in natsorted(kernel.arg_dict):
                lines.append(str(kernel.arg_dict[arg_name]))

        if "domains" in what:
            lines.extend(sep)
            if show_labels:
                lines.append("DOMAINS:")
            for dom, parents in zip(kernel.domains, kernel.all_parents_per_domain()):
                lines.append(len(parents)*"  " + str(dom))

        if "tags" in what:
            lines.extend(sep)
            if show_labels:
                lines.append("INAME IMPLEMENTATION TAGS:")
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
                    key=lambda tv: tv.name):
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

        if "schedule" in what and kernel.schedule is not None:
            lines.extend(sep)
            if show_labels:
                lines.append("LINEARIZATION:")
            from loopy.schedule import dump_schedule
            lines.append(dump_schedule(kernel, kernel.schedule))

        lines.extend(sep)

        return "\n".join(lines)

    def __str__(self):
        return self.stringify()

    def __unicode__(self):
        return self.stringify()

    # }}}

    # {{{ implementation arguments

    @property
    @memoize_method
    def impl_arg_to_arg(self):
        from loopy.kernel.array import ArrayBase

        result = {}

        for arg in self.args:
            if not isinstance(arg, ArrayBase):
                result[arg.name] = arg
                continue

            if arg.shape is None or arg.dim_tags is None:
                result[arg.name] = arg
                continue

            subscripts_and_names = arg.subscripts_and_names()
            if subscripts_and_names is None:
                result[arg.name] = arg
                continue

            for index, sub_arg_name in subscripts_and_names:
                result[sub_arg_name] = arg

        return result

    # }}}

    # {{{ handle linearization variable that doesn't yet exist

    @property
    def linearization(self):
        return self.schedule

    # }}}

    # {{{ direct execution

    def __call__(self, *args, **kwargs):
        """
        Execute the :class:`LoopKernel`.
        """
        warn("Calling a LoopKernel is deprecated, call a TranslationUnit "
                "instead.", DeprecationWarning, stacklevel=2)
        from loopy.program import make_program
        program = make_program(self)
        return program(*args, **kwargs)

    # }}}

    # {{{ pickling

    def __getstate__(self):
        result = {
                key: getattr(self, key)
                for key in self.__class__.fields
                if hasattr(self, key)}

        result.pop("cache_manager", None)

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

        return (result, self._pytools_persistent_hash_digest)

    def __setstate__(self, state):
        attribs, p_hash_digest = state

        new_fields = set()

        for k, v in attribs.items():
            setattr(self, k, v)
            new_fields.add(k)

        self.register_fields(new_fields)

        if 0:
            # {{{ check that 'reconstituted' object has same hash

            from loopy.tools import LoopyKeyBuilder
            assert p_hash_digest == LoopyKeyBuilder()(self)

            # }}}

        self._pytools_persistent_hash_digest = p_hash_digest

        from loopy.kernel.tools import SetOperationCacheManager
        self.cache_manager = SetOperationCacheManager()
        self._kernel_executor_cache = {}

    # }}}

    # {{{ persistent hash key generation / comparison

    hash_fields = (
            "domains",
            "instructions",
            "args",
            "schedule",
            "name",
            "preambles",
            "assumptions",
            "local_sizes",
            "temporary_variables",
            "inames",
            "substitutions",
            "iname_slab_increments",
            "loop_priority",
            "silenced_warnings",
            "options",
            "state",
            "target",
            )

    comparison_fields = hash_fields + (
            # Contains pymbolic expressions, hence a (small) headache to hash.
            # Likely not needed for hash uniqueness => headache avoided.
            "applied_iname_rewrites",

            # These are lists of functions. It's not clear how to
            # hash these correctly, so let's not attempt it. We'll
            # just assume that the rest of the hash is specific enough
            # that we won't have to rely on differences in these to
            # resolve hash conflicts.

            "preamble_generators",
            "symbol_manglers",
            )

    update_persistent_hash = update_persistent_hash

    @memoize_method
    def __hash__(self):
        from loopy.tools import LoopyKeyBuilder
        import hashlib
        key_hash = hashlib.sha256()
        self.update_persistent_hash(key_hash, LoopyKeyBuilder())
        return hash(key_hash.digest())

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, LoopKernel):
            return False

        for field_name in self.comparison_fields:
            if field_name == "domains":
                if len(self.domains) != len(other.domains):
                    return False

                for set_a, set_b in zip(self.domains, other.domains):
                    if not (set_a.plain_is_equal(set_b) or set_a.is_equal(set_b)):
                        return False

            elif field_name == "assumptions":
                if not (
                        self.assumptions.plain_is_equal(other.assumptions)
                        or self.assumptions.is_equal(other.assumptions)):
                    return False

            elif getattr(self, field_name) != getattr(other, field_name):
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    # }}}

    def copy(self, **kwargs):
        if "iname_to_tags" in kwargs:
            if "inames" in kwargs:
                raise LoopyError("Cannot pass both `inames` and `iname_to_tags` to "
                        "LoopKernel.copy")

            kwargs["inames"] = None

        # Avoid carrying over an invalid cache when other parts of the kernel
        # are modified.
        kwargs["_cached_written_variables"] = None

        return super().copy(**kwargs)

# }}}

# vim: foldmethod=marker

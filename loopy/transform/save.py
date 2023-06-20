__copyright__ = "Copyright (C) 2016 Matt Wala"

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

from immutables import Map

from loopy.diagnostic import LoopyError
import loopy as lp

from loopy.kernel.data import auto, AddressSpace
from pytools import memoize_method, Record
from loopy.kernel.data import Iname
from loopy.schedule import (
            EnterLoop, LeaveLoop, RunInstruction,
            CallKernel, ReturnFromKernel, Barrier)

from loopy.schedule.tools import get_block_boundaries


import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. currentmodule:: loopy

.. autofunction:: save_and_reload_temporaries
"""


# {{{ liveness analysis

class LivenessResult(dict):

    class InstructionResult(Record):
        __slots__ = ["live_in", "live_out"]

    @classmethod
    def make_empty(cls, nscheditems):
        return cls((idx, cls.InstructionResult(live_in=set(), live_out=set()))
                   for idx in range(nscheditems))


class LivenessAnalysis:

    def __init__(self, kernel):
        self.kernel = kernel
        self.schedule = kernel.linearization

    @memoize_method
    def get_successor_relation(self):
        successors = {}
        block_bounds = get_block_boundaries(self.kernel.linearization)

        for idx, (item, next_item) in enumerate(zip(
                reversed(self.schedule),
                reversed(self.schedule + [None]))):
            sched_idx = len(self.schedule) - idx - 1

            # Look at next_item
            if next_item is None:
                after = set()
            elif isinstance(next_item, EnterLoop):
                # Account for empty loop
                loop_end = block_bounds[sched_idx + 1]
                after = successors[loop_end] | {sched_idx + 1}
            elif isinstance(next_item, (LeaveLoop, RunInstruction,
                    CallKernel, ReturnFromKernel, Barrier)):
                after = {sched_idx + 1}
            else:
                raise LoopyError("unexpected type of schedule item: {ty}"
                    .format(ty=type(next_item).__name__))

            # Look at item
            if isinstance(item, LeaveLoop):
                # Account for loop
                loop_begin = block_bounds[sched_idx]
                after |= {loop_begin}
            elif not isinstance(item, (EnterLoop, RunInstruction,
                    CallKernel, ReturnFromKernel, Barrier)):
                raise LoopyError("unexpected type of schedule item: {ty}"
                    .format(ty=type(item).__name__))

            successors[sched_idx] = after

        return successors

    def get_gen_and_kill_sets(self):
        gen = {idx: set() for idx in range(len(self.schedule))}
        kill = {idx: set() for idx in range(len(self.schedule))}

        for sched_idx, sched_item in enumerate(self.schedule):
            if not isinstance(sched_item, RunInstruction):
                continue
            insn = self.kernel.id_to_insn[sched_item.insn_id]
            for var in insn.assignee_var_names():
                if var not in self.kernel.temporary_variables:
                    continue
                if not insn.predicates:
                    # Fully kills the liveness only when unconditional.
                    kill[sched_idx].add(var)
                if len(self.kernel.temporary_variables[var].shape) > 0:
                    # For an array variable, all definitions generate a use as
                    # well, because the write could be a partial write,
                    # necessitating a reload of whatever is not written.
                    #
                    # We don't currently check if the write is a partial write
                    # or a full write. Instead, we analyze the access
                    # footprint later on to determine how much to reload/save.
                    gen[sched_idx].add(var)
            for var in insn.read_dependency_names():
                if var not in self.kernel.temporary_variables:
                    continue
                gen[sched_idx].add(var)

        return gen, kill

    @memoize_method
    def liveness(self):
        logger.info("running liveness analysis")
        successors = self.get_successor_relation()
        gen, kill = self.get_gen_and_kill_sets()

        # Fixed point iteration for liveness analysis
        lr = LivenessResult.make_empty(len(self.schedule))

        prev_lr = None

        while prev_lr != lr:
            from copy import deepcopy
            prev_lr = deepcopy(lr)
            for idx in range(len(self.schedule) - 1, -1, -1):
                for succ in successors[idx]:
                    lr[idx].live_out.update(lr[succ].live_in)
                lr[idx].live_in = gen[idx] | (lr[idx].live_out - kill[idx])

        logger.info("done running liveness analysis")

        return lr

    def print_liveness(self):
        print(75 * "-")
        print("LIVE IN:")
        for sched_idx in range(len(self.schedule)):
            print("{item}: {{{vars}}}".format(
                item=sched_idx,
                vars=", ".join(sorted(self[sched_idx].live_in))))
        print(75 * "-")
        print("LIVE OUT:")
        for sched_idx in range(len(self.schedule)):
            print("{item}: {{{vars}}}".format(
                item=sched_idx,
                vars=", ".join(sorted(self[sched_idx].live_out))))
        print(75 * "-")

    def __getitem__(self, sched_idx):
        """
        :arg insn: An instruction name or instance of
            :class:`loopy.instruction.InstructionBase`

        :returns: A :class:`LivenessResult` associated with `insn`
        """
        return self.liveness()[sched_idx]

# }}}


# {{{ save and reload implementation

class TemporarySaver:

    class PromotedTemporary(Record):
        """
        .. attribute:: name

            The name of the new temporary.

        .. attribute:: orig_temporary_name

            The name of original temporary variable object.

        .. attribute:: hw_dims

            A list of expressions, to be added in front of the shape
            of the promoted temporary value, corresponding to
            hardware dimensions

        .. attribute:: hw_tags

            The tags for the inames associated with hw_dims

        .. attribute:: non_hw_dims

            A list of expressions, to be added in front of the shape
            of the promoted temporary value, corresponding to
            non-hardware dimensions
        """

        __slots__ = """
                name
                orig_temporary_name
                hw_dims
                hw_tags
                non_hw_dims""".split()

        def as_kernel_temporary(self, kernel):
            temporary = kernel.temporary_variables[self.orig_temporary_name]
            from loopy.kernel.data import TemporaryVariable
            return TemporaryVariable(
                name=self.name,
                dtype=temporary.dtype,
                address_space=AddressSpace.GLOBAL,
                shape=self.new_shape)

        @property
        def new_shape(self):
            return self.hw_dims + self.non_hw_dims

    def __init__(self, kernel, callables_table):
        self.kernel = kernel
        self.callables_table = callables_table
        self.var_name_gen = kernel.get_var_name_generator()
        self.insn_name_gen = kernel.get_instruction_id_generator()

        # These fields keep track of updates to the kernel.
        from collections import defaultdict
        self.insns_to_insert = []
        self.insns_to_update = {}
        self.updated_iname_objs = Map()
        self.updated_temporary_variables = {}

        # temporary name -> save or reload insn ids
        self.temporary_to_save_ids = defaultdict(set)
        self.temporary_to_reload_ids = defaultdict(set)
        self.subkernel_to_newly_added_insn_ids = defaultdict(set)

        # Maps names of base_storage to the name of the temporary
        # representative chosen for saves/reloads
        self.base_storage_to_representative = {}

        from loopy.kernel.data import ValueArg
        import islpy as isl
        self.new_subdomain = (
                isl.BasicSet.universe(
                    isl.Space.create_from_names(
                        isl.DEFAULT_CONTEXT,
                        set=[],
                        params={
                            arg.name for arg in kernel.args
                            if isinstance(arg, ValueArg)})))

    def find_accessing_instructions_in_subkernel(self, temporary, subkernel):
        # Find all accessing instructions in the subkernel. If base_storage is
        # present, this includes instructions that access aliasing memory.

        aliasing_names = {temporary}
        base_storage = self.kernel.temporary_variables[temporary].base_storage

        if base_storage is not None:
            aliasing_names |= self.base_storage_to_temporary_map[base_storage]

        from loopy.kernel.tools import get_subkernel_to_insn_id_map
        accessing_insns_in_subkernel = set()
        subkernel_insns = get_subkernel_to_insn_id_map(self.kernel)[subkernel]

        for name in aliasing_names:
            try:
                accessing_insns_in_subkernel |= (
                        self.kernel.reader_map()[name] & subkernel_insns)
            except KeyError:
                pass

            try:
                accessing_insns_in_subkernel |= (
                        self.kernel.writer_map()[name] & subkernel_insns)
            except KeyError:
                pass

        return frozenset(accessing_insns_in_subkernel)

    @cached_property
    def base_storage_to_temporary_map(self):
        from collections import defaultdict

        result = defaultdict(set)

        for temporary in self.kernel.temporary_variables.values():
            if temporary.base_storage is None:
                continue
            result[temporary.base_storage].add(temporary.name)

        return result

    @cached_property
    def subkernel_to_slice_indices(self):
        result = {}

        for sched_item_idx, sched_item in enumerate(self.kernel.linearization):
            if isinstance(sched_item, CallKernel):
                start_idx = sched_item_idx
            elif isinstance(sched_item, ReturnFromKernel):
                result[sched_item.kernel_name] = (start_idx, 1 + sched_item_idx)

        return result

    @cached_property
    def subkernel_to_surrounding_inames(self):
        current_outer_inames = set()
        within_subkernel = False
        result = {}

        for sched_item in self.kernel.linearization:
            if isinstance(sched_item, CallKernel):
                within_subkernel = True
                result[sched_item.kernel_name] = frozenset(current_outer_inames)
            elif isinstance(sched_item, ReturnFromKernel):
                within_subkernel = False
            elif isinstance(sched_item, EnterLoop):
                if not within_subkernel:
                    current_outer_inames.add(sched_item.iname)
            elif isinstance(sched_item, LeaveLoop):
                current_outer_inames.discard(sched_item.iname)

        return result

    @memoize_method
    def get_enclosing_global_barrier_pair(self, subkernel):
        subkernel_start, subkernel_end = (
            self.subkernel_to_slice_indices[subkernel])

        def is_global_barrier(item):
            return isinstance(item, Barrier) and \
                item.synchronization_kind == "global"

        try:
            pre_barrier = next(item for item in
                self.kernel.linearization[subkernel_start::-1]
                if is_global_barrier(item)).originating_insn_id
        except StopIteration:
            pre_barrier = None

        try:
            post_barrier = next(item for item in
                self.kernel.linearization[subkernel_end:]
                if is_global_barrier(item)).originating_insn_id
        except StopIteration:
            post_barrier = None

        return (pre_barrier, post_barrier)

    def get_hw_axis_sizes_and_tags_for_save_slot(self, temporary):
        """
        This is used for determining the amount of global storage needed for saving
        and restoring the temporary across kernel calls, due to hardware
        parallel inames (the inferred axes get prefixed to the number of
        dimensions in the temporary).

        In the case of local temporaries, inames that are tagged
        hw-local do not contribute to the global storage shape.
        """
        accessor_insn_ids = frozenset(
            self.kernel.reader_map()[temporary.name]
            | self.kernel.writer_map()[temporary.name])

        group_tags = None
        local_tags = None

        def _sortedtags(tags):
            return sorted(tags, key=lambda tag: tag.axis)

        for insn_id in accessor_insn_ids:
            insn = self.kernel.id_to_insn[insn_id]

            my_group_tags = []
            my_local_tags = []

            for iname in insn.within_inames:
                tags = self.kernel.iname_tags(iname)

                if not tags:
                    continue

                from loopy.kernel.data import (GroupInameTag, LocalInameTag,
                        ConcurrentTag, filter_iname_tags_by_type)

                if filter_iname_tags_by_type(tags, GroupInameTag):
                    tag, = filter_iname_tags_by_type(tags, GroupInameTag, 1)
                    my_group_tags.append(tag)
                elif filter_iname_tags_by_type(tags, LocalInameTag):
                    tag, = filter_iname_tags_by_type(tags, LocalInameTag, 1)
                    my_local_tags.append(tag)
                elif filter_iname_tags_by_type(tags, ConcurrentTag):
                    raise LoopyError(
                        "iname '%s' is tagged with '%s' - only "
                        "group and local tags are supported for "
                        "auto save/reload of temporaries" %
                        (iname, tags))

            if group_tags is None:
                group_tags = _sortedtags(my_group_tags)
                local_tags = _sortedtags(my_local_tags)
                group_tags_originating_insn_id = insn_id

            if (
                    group_tags != _sortedtags(my_group_tags)
                    or local_tags != _sortedtags(my_local_tags)):
                raise LoopyError(
                    "inconsistent parallel tags across instructions that access "
                    "'%s' (specifically, instruction '%s' has tags '%s' but "
                    "instruction '%s' has tags '%s')"
                    % (temporary.name,
                       group_tags_originating_insn_id, group_tags + local_tags,
                       insn_id, my_group_tags + my_local_tags))

        if group_tags is None:
            assert local_tags is None
            return (), ()

        group_sizes, local_sizes = (
            self.kernel.get_grid_sizes_for_insn_ids_as_exprs(accessor_insn_ids,
                self.callables_table))

        if temporary.address_space == lp.AddressSpace.LOCAL:
            # Elide local axes in the save slot for local temporaries.
            del local_tags[:]
            local_sizes = ()

        # We set hw_dims to be arranged according to the order:
        #    g.0 < g.1 < ... < l.0 < l.1 < ...
        return (group_sizes + local_sizes), tuple(group_tags + local_tags)

    @memoize_method
    def auto_promote_temporary(self, temporary_name):
        temporary = self.kernel.temporary_variables[temporary_name]

        if temporary.address_space == AddressSpace.GLOBAL:
            # Nothing to be done for global temporaries (I hope)
            return None

        if temporary.initializer is not None:
            # Temporaries with initializers do not need saving/reloading - the
            # code generation takes care of emitting the initializers.
            assert temporary.read_only
            return None

        base_storage_conflict = (
            self.base_storage_to_representative.get(
                temporary.base_storage, temporary) is not temporary)

        if base_storage_conflict:
            raise NotImplementedError(
                "tried to save/reload multiple temporaries with the "
                "same base_storage; this is currently not supported")

        hw_dims, hw_tags = self.get_hw_axis_sizes_and_tags_for_save_slot(temporary)
        non_hw_dims = temporary.shape

        if len(non_hw_dims) == 0 and len(hw_dims) == 0:
            # Scalar not in hardware: ensure at least one dimension.
            non_hw_dims = (1,)

        backing_temporary = self.PromotedTemporary(
            name=self.var_name_gen(temporary.name + "_save_slot"),
            orig_temporary_name=temporary.name,
            hw_dims=hw_dims,
            hw_tags=hw_tags,
            non_hw_dims=non_hw_dims)

        if temporary.base_storage is not None:
            self.base_storage_to_representative[temporary.base_storage] = (
                    backing_temporary)

        return backing_temporary

    def save_or_reload_impl(self, temporary, subkernel, mode,
                             promoted_temporary=lp.auto):
        assert mode in ("save", "reload")

        if promoted_temporary is auto:
            promoted_temporary = self.auto_promote_temporary(temporary)

        if promoted_temporary is None:
            return

        new_subdomain, hw_inames, dim_inames, iname_objs = (
            self.augment_domain_for_save_or_reload(
                self.new_subdomain, promoted_temporary, mode, subkernel))

        self.new_subdomain = new_subdomain

        save_or_load_insn_id = self.insn_name_gen(
            f"{temporary}.{mode}")

        def add_subscript_if_subscript_nonempty(agg, subscript=()):
            from pymbolic.primitives import Subscript, Variable
            if len(subscript) == 0:
                return Variable(agg)
            else:
                return Subscript(
                    Variable(agg),
                    tuple(map(Variable, subscript)))

        orig_temporary = (
            self.kernel.temporary_variables[
                promoted_temporary.orig_temporary_name])
        dim_inames_trunc = dim_inames[:len(orig_temporary.shape)]

        args = (
            add_subscript_if_subscript_nonempty(
                temporary, subscript=dim_inames_trunc),
            add_subscript_if_subscript_nonempty(
                promoted_temporary.name, subscript=hw_inames + dim_inames))

        if mode == "save":
            args = reversed(args)

        accessing_insns_in_subkernel = self.find_accessing_instructions_in_subkernel(
                temporary, subkernel)

        if mode == "save":
            depends_on = accessing_insns_in_subkernel
            update_deps = frozenset()
        elif mode == "reload":
            depends_on = frozenset()
            update_deps = accessing_insns_in_subkernel

        pre_barrier, post_barrier = self.get_enclosing_global_barrier_pair(subkernel)

        if pre_barrier is not None:
            depends_on |= {pre_barrier}

        if post_barrier is not None:
            update_deps |= {post_barrier}

        # Create the load / store instruction.
        from loopy.kernel.data import Assignment
        save_or_load_insn = Assignment(
            *args,
            id=save_or_load_insn_id,
            within_inames=(
                self.subkernel_to_surrounding_inames[subkernel]
                | frozenset(hw_inames + dim_inames)),
            within_inames_is_final=True,
            depends_on=depends_on)

        if mode == "save":
            self.temporary_to_save_ids[temporary].add(save_or_load_insn_id)
        else:
            self.temporary_to_reload_ids[temporary].add(save_or_load_insn_id)

        self.subkernel_to_newly_added_insn_ids[subkernel].add(save_or_load_insn_id)

        self.insns_to_insert.append(save_or_load_insn)

        for insn_id in update_deps:
            insn = self.insns_to_update.get(insn_id, self.kernel.id_to_insn[insn_id])
            self.insns_to_update[insn_id] = insn.copy(
                depends_on=insn.depends_on | frozenset([save_or_load_insn_id]))

        self.updated_temporary_variables[promoted_temporary.name] = (
            promoted_temporary.as_kernel_temporary(self.kernel))

        self.updated_iname_objs = self.updated_iname_objs.update(iname_objs)

    @memoize_method
    def finish(self):
        new_instructions = []

        insns_to_insert = {insn.id: insn for insn in self.insns_to_insert}

        for orig_insn in self.kernel.instructions:
            if orig_insn.id in self.insns_to_update:
                new_instructions.append(self.insns_to_update[orig_insn.id])
            else:
                new_instructions.append(orig_insn)
        new_instructions.extend(
            sorted(insns_to_insert.values(), key=lambda insn: insn.id))

        self.updated_iname_objs = self.updated_iname_objs.update(self.kernel.inames)
        self.updated_temporary_variables.update(self.kernel.temporary_variables)

        new_domains = list(self.kernel.domains)
        import islpy as isl
        if self.new_subdomain.dim(isl.dim_type.set) > 0:
            new_domains.append(self.new_subdomain)

        kernel = self.kernel.copy(
            domains=new_domains,
            instructions=new_instructions,
            inames=self.updated_iname_objs,
            temporary_variables=self.updated_temporary_variables,
            overridden_get_grid_sizes_for_insn_ids=None)

        # Add nosync directives to any saves or reloads that were added with a
        # potential dependency chain.
        from loopy.kernel.tools import get_subkernels
        for subkernel in get_subkernels(kernel):
            relevant_insns = self.subkernel_to_newly_added_insn_ids[subkernel]

            from itertools import product
            for temporary in self.temporary_to_reload_ids:
                for source, sink in product(
                        relevant_insns & self.temporary_to_reload_ids[temporary],
                        relevant_insns & self.temporary_to_save_ids[temporary]):
                    kernel = lp.add_nosync(kernel, "global", source, sink)

        from loopy.kernel.tools import assign_automatic_axes
        return assign_automatic_axes(kernel, self.callables_table)

    def save(self, temporary, subkernel):
        self.save_or_reload_impl(temporary, subkernel, "save")

    def reload(self, temporary, subkernel):
        self.save_or_reload_impl(temporary, subkernel, "reload")

    def augment_domain_for_save_or_reload(self,
            domain, promoted_temporary, mode, subkernel):
        """
        Add new axes to the domain corresponding to the dimensions of
        `promoted_temporary`. These axes will be used in the save/
        reload stage. These get prefixed onto the already existing axes.
        """
        assert mode in ("save", "reload")
        import islpy as isl

        orig_temporary = (
                self.kernel.temporary_variables[
                    promoted_temporary.orig_temporary_name])
        orig_dim = domain.dim(isl.dim_type.set)

        # Tags for newly added inames
        iname_objs = {}

        from loopy.symbolic import aff_from_expr

        # FIXME: Restrict size of new inames to access footprint.

        # Add dimension-dependent inames.
        dim_inames = []
        domain = domain.add_dims(isl.dim_type.set,
                            len(promoted_temporary.non_hw_dims)
                            + len(promoted_temporary.hw_dims))

        for dim_idx, dim_size in enumerate(promoted_temporary.non_hw_dims):
            new_iname = self.insn_name_gen("{name}_{mode}_axis_{dim}_{sk}".
                format(name=orig_temporary.name,
                       mode=mode,
                       dim=dim_idx,
                       sk=subkernel))
            domain = domain.set_dim_name(
                isl.dim_type.set, orig_dim + dim_idx, new_iname)

            if orig_temporary.address_space == AddressSpace.LOCAL:
                # If the temporary has local scope, then loads / stores can
                # be done in parallel.
                from loopy.kernel.data import AutoFitLocalInameTag
                iname_objs[new_iname] = Iname(
                        new_iname, tags=frozenset([AutoFitLocalInameTag()]))

            dim_inames.append(new_iname)

            # Add size information.
            aff = isl.affs_from_space(domain.space)
            domain &= aff[0].le_set(aff[new_iname])
            domain &= aff[new_iname].lt_set(aff_from_expr(domain.space, dim_size))

        dim_offset = orig_dim + len(promoted_temporary.non_hw_dims)

        hw_inames = []
        # Add hardware dims.
        for hw_iname_idx, (hw_tag, dim) in enumerate(
                zip(promoted_temporary.hw_tags, promoted_temporary.hw_dims)):
            new_iname = self.insn_name_gen("{name}_{mode}_hw_dim_{dim}_{sk}".
                format(name=orig_temporary.name,
                       mode=mode,
                       dim=hw_iname_idx,
                       sk=subkernel))
            domain = domain.set_dim_name(
                isl.dim_type.set, dim_offset + hw_iname_idx, new_iname)

            aff = isl.affs_from_space(domain.space)
            domain = (domain
                &
                aff[0].le_set(aff[new_iname])
                &
                aff[new_iname].lt_set(aff_from_expr(domain.space, dim)))

            self.updated_iname_objs = self.updated_iname_objs.set(new_iname,
                    Iname(name=new_iname, tags=frozenset([hw_tag])))
            hw_inames.append(new_iname)

        # The operations on the domain above return a Set object, but the
        # underlying domain should be expressible as a single BasicSet.
        domain_list = domain.get_basic_set_list()
        assert domain_list.n_basic_set() == 1
        domain = domain_list.get_basic_set(0)
        return domain, hw_inames, dim_inames, iname_objs

# }}}


# {{{ auto save and reload across kernel calls

def save_and_reload_temporaries(program, entrypoint=None):
    """
    Add instructions to save and reload temporary variables that are live
    across kernel calls.

    The basic code transformation turns schedule segments::

        t = <...>
        <return followed by call>
        <...> = t

    into this code::

        t = <...>
        t_save_slot = t
        <return followed by call>
        t = t_save_slot
        <...> = t

    where `t_save_slot` is a newly-created global temporary variable.

    :returns: The resulting kernel
    """
    if entrypoint is None:
        if len(program.entrypoints) != 1:
            raise LoopyError("Missing argument 'entrypoint'.")
        entrypoint = list(program.entrypoints)[0]

    knl = program[entrypoint]

    if not knl.linearization:
        program = lp.preprocess_program(program)
        from loopy.schedule import get_one_linearized_kernel
        knl = get_one_linearized_kernel(program[entrypoint],
                program.callables_table)

    assert knl.linearization is not None

    liveness = LivenessAnalysis(knl)
    saver = TemporarySaver(knl, program.callables_table)

    from loopy.schedule.tools import (
        temporaries_read_in_subkernel, temporaries_written_in_subkernel)

    for sched_idx, sched_item in enumerate(knl.linearization):

        if isinstance(sched_item, CallKernel):
            # Any written temporary that is live-out needs to be read into
            # memory because of the potential for partial writes.
            if sched_idx == 0:
                # Kernel entry: nothing live
                interesting_temporaries = set()
            else:
                subkernel = sched_item.kernel_name
                interesting_temporaries = (
                    temporaries_read_in_subkernel(knl, subkernel)
                    | temporaries_written_in_subkernel(knl,
                                                       subkernel))

            for temporary in liveness[sched_idx].live_out & interesting_temporaries:
                logger.info("reloading {} at entry of {}"
                        .format(temporary, sched_item.kernel_name))
                saver.reload(temporary, sched_item.kernel_name)

        elif isinstance(sched_item, ReturnFromKernel):
            if sched_idx == len(knl.linearization) - 1:
                # Kernel exit: nothing live
                interesting_temporaries = set()
            else:
                subkernel = sched_item.kernel_name
                interesting_temporaries = (
                    temporaries_written_in_subkernel(knl, subkernel))

            for temporary in liveness[sched_idx].live_in & interesting_temporaries:
                logger.info("saving {} before return of {}"
                        .format(temporary, sched_item.kernel_name))
                saver.save(temporary, sched_item.kernel_name)

    return program.with_kernel(saver.finish())

# }}}


# vim: foldmethod=marker

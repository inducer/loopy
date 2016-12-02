from __future__ import division, absolute_import

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


from loopy.diagnostic import LoopyError
import loopy as lp

from loopy.kernel.data import auto
from pytools import memoize_method, Record
from loopy.schedule import (
            EnterLoop, LeaveLoop, RunInstruction,
            CallKernel, ReturnFromKernel, Barrier)

from loopy.schedule.tools import (get_block_boundaries, InstructionQuery)


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


class LivenessAnalysis(object):

    def __init__(self, kernel):
        self.kernel = kernel
        self.schedule = self.kernel.schedule

    @memoize_method
    def get_successor_relation(self):
        successors = {}
        block_bounds = get_block_boundaries(self.kernel.schedule)

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
                after = successors[loop_end] | set([sched_idx + 1])
            elif isinstance(next_item, (LeaveLoop, RunInstruction,
                    CallKernel, ReturnFromKernel, Barrier)):
                after = set([sched_idx + 1])
            else:
                raise LoopyError("unexpected type of schedule item: {ty}"
                    .format(ty=type(next_item).__name__))

            # Look at item
            if isinstance(item, LeaveLoop):
                # Account for loop
                loop_begin = block_bounds[sched_idx]
                after |= set([loop_begin])
            elif not isinstance(item, (EnterLoop, RunInstruction,
                    CallKernel, ReturnFromKernel, Barrier)):
                raise LoopyError("unexpected type of schedule item: {ty}"
                    .format(ty=type(item).__name__))

            successors[sched_idx] = after

        return successors

    def get_gen_and_kill_sets(self):
        gen = dict((idx, set()) for idx in range(len(self.schedule)))
        kill = dict((idx, set()) for idx in range(len(self.schedule)))

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
        logging.info("running liveness analysis")
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

        logging.info("done running liveness analysis")

        return lr

    def print_liveness(self):
        print(75 * "-")
        print("LIVE IN:")
        for sched_idx, sched_item in enumerate(self.schedule):
            print("{item}: {{{vars}}}".format(
                item=sched_idx,
                vars=", ".join(sorted(self[sched_idx].live_in))))
        print(75 * "-")
        print("LIVE OUT:")
        for sched_idx, sched_item in enumerate(self.schedule):
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

class TemporarySaver(object):

    class PromotedTemporary(Record):
        """
        .. attribute:: name

            The name of the new temporary.

        .. attribute:: orig_temporary

            The original temporary variable object.

        .. attribute:: hw_inames

            The common list of hw axes that define the original object.

        .. attribute:: hw_dims

            A list of expressions, to be added in front of the shape
            of the promoted temporary value, corresponding to
            hardware dimensions

        .. attribute:: non_hw_dims

            A list of expressions, to be added in front of the shape
            of the promoted temporary value, corresponding to
            non-hardware dimensions
        """

        @memoize_method
        def as_variable(self):
            temporary = self.orig_temporary
            from loopy.kernel.data import TemporaryVariable, temp_var_scope
            return TemporaryVariable(
                name=self.name,
                dtype=temporary.dtype,
                scope=temp_var_scope.GLOBAL,
                shape=self.new_shape)

        @property
        def new_shape(self):
            return self.hw_dims + self.non_hw_dims

    def __init__(self, kernel):
        self.kernel = kernel
        self.insn_query = InstructionQuery(kernel)
        self.var_name_gen = kernel.get_var_name_generator()
        self.insn_name_gen = kernel.get_instruction_id_generator()
        # These fields keep track of updates to the kernel.
        self.insns_to_insert = []
        self.insns_to_update = {}
        self.extra_args_to_add = {}
        self.updated_iname_to_tag = {}
        self.updated_temporary_variables = {}
        self.saves_or_reloads_added = {}

    @memoize_method
    def auto_promote_temporary(self, temporary_name):
        temporary = self.kernel.temporary_variables[temporary_name]

        from loopy.kernel.data import temp_var_scope
        if temporary.scope == temp_var_scope.GLOBAL:
            # Nothing to be done for global temporaries (I hope)
            return None

        if temporary.base_storage is not None:
            raise ValueError(
                "Cannot promote temporaries with base_storage to global")

        # `hw_inames`: The set of hw-parallel tagged inames that this temporary
        # is associated with. This is used for determining the shape of the
        # global storage needed for saving and restoring the temporary across
        # kernel calls.
        #
        # TODO: Make a policy decision about which dimensions to use. Currently,
        # the code looks at each instruction that defines or uses the temporary,
        # and takes the common set of hw-parallel tagged inames associated with
        # these instructions.
        #
        # Furthermore, in the case of local temporaries, inames that are tagged
        # hw-local do not contribute to the global storage shape.
        hw_inames = self.insn_query.common_hw_inames(
            self.insn_query.insns_reading_or_writing(temporary.name))

        # We want hw_inames to be arranged according to the order:
        #    g.0 < g.1 < ... < l.0 < l.1 < ...
        # Sorting lexicographically accomplishes this.
        hw_inames = sorted(hw_inames,
            key=lambda iname: str(self.kernel.iname_to_tag[iname]))

        # Calculate the sizes of the dimensions that get added in front for
        # the global storage of the temporary.
        hw_dims = []

        backing_hw_inames = []

        for iname in hw_inames:
            tag = self.kernel.iname_to_tag[iname]
            from loopy.kernel.data import LocalIndexTag
            is_local_iname = isinstance(tag, LocalIndexTag)
            if is_local_iname and temporary.scope == temp_var_scope.LOCAL:
                # Restrict shape to that of group inames for locals.
                continue
            backing_hw_inames.append(iname)
            from loopy.isl_helpers import static_max_of_pw_aff
            from loopy.symbolic import aff_to_expr
            hw_dims.append(
                aff_to_expr(
                    static_max_of_pw_aff(
                        self.kernel.get_iname_bounds(iname).size, False)))

        non_hw_dims = temporary.shape

        if len(non_hw_dims) == 0 and len(hw_dims) == 0:
            # Scalar not in hardware: ensure at least one dimension.
            non_hw_dims = (1,)

        backing_temporary = self.PromotedTemporary(
            name=self.var_name_gen(temporary.name + "_save_slot"),
            orig_temporary=temporary,
            hw_dims=tuple(hw_dims),
            non_hw_dims=non_hw_dims,
            hw_inames=backing_hw_inames)

        return backing_temporary

    def save_or_reload_impl(self, temporary, subkernel, mode,
                             promoted_temporary=lp.auto):
        assert mode in ("save", "reload")

        if promoted_temporary is auto:
            promoted_temporary = self.auto_promote_temporary(temporary)

        if promoted_temporary is None:
            return

        from loopy.kernel.tools import DomainChanger
        dchg = DomainChanger(
            self.kernel,
            frozenset(
                self.insn_query.inames_in_subkernel(subkernel) |
                set(promoted_temporary.hw_inames)))

        domain, hw_inames, dim_inames, iname_to_tag = \
            self.augment_domain_for_save_or_reload(
                dchg.domain, promoted_temporary, mode, subkernel)

        self.kernel = dchg.get_kernel_with(domain)

        save_or_load_insn_id = self.insn_name_gen(
            "{name}.{mode}".format(name=temporary, mode=mode))

        def subscript_or_var(agg, subscript=()):
            from pymbolic.primitives import Subscript, Variable
            if len(subscript) == 0:
                return Variable(agg)
            else:
                return Subscript(
                    Variable(agg),
                    tuple(map(Variable, subscript)))

        dim_inames_trunc = dim_inames[:len(promoted_temporary.orig_temporary.shape)]

        args = (
            subscript_or_var(
                temporary, dim_inames_trunc),
            subscript_or_var(
                promoted_temporary.name, hw_inames + dim_inames))

        if mode == "save":
            args = reversed(args)

        accessing_insns_in_subkernel = (
            self.insn_query.insns_reading_or_writing(temporary) &
            self.insn_query.insns_in_subkernel(subkernel))

        if mode == "save":
            depends_on = accessing_insns_in_subkernel
            update_deps = frozenset()
        elif mode == "reload":
            depends_on = frozenset()
            update_deps = accessing_insns_in_subkernel

        pre_barrier, post_barrier = self.insn_query.pre_and_post_barriers(subkernel)

        if pre_barrier is not None:
            depends_on |= set([pre_barrier])

        if post_barrier is not None:
            update_deps |= set([post_barrier])

        # Create the load / store instruction.
        from loopy.kernel.data import Assignment
        save_or_load_insn = Assignment(
            *args,
            id=save_or_load_insn_id,
            within_inames=(
                self.insn_query.inames_in_subkernel(subkernel) |
                frozenset(hw_inames + dim_inames)),
            within_inames_is_final=True,
            depends_on=depends_on,
            boostable=False,
            boostable_into=frozenset())

        if temporary not in self.saves_or_reloads_added:
            self.saves_or_reloads_added[temporary] = set()
        self.saves_or_reloads_added[temporary].add(save_or_load_insn_id)

        self.insns_to_insert.append(save_or_load_insn)

        for insn_id in update_deps:
            insn = self.insns_to_update.get(insn_id, self.kernel.id_to_insn[insn_id])
            self.insns_to_update[insn_id] = insn.copy(
                depends_on=insn.depends_on | frozenset([save_or_load_insn_id]))

        self.updated_temporary_variables[promoted_temporary.name] = \
            promoted_temporary.as_variable()

        self.updated_iname_to_tag.update(iname_to_tag)

    @memoize_method
    def finish(self):
        new_instructions = []

        insns_to_insert = dict((insn.id, insn) for insn in self.insns_to_insert)

        # Add global no_sync_with between any added reloads and saves
        from six import iteritems
        for temporary, added_insns in iteritems(self.saves_or_reloads_added):
            for insn_id in added_insns:
                insn = insns_to_insert[insn_id]
                insns_to_insert[insn_id] = insn.copy(
                    no_sync_with=frozenset(
                        (added_insn, "global") for added_insn in added_insns))

        for orig_insn in self.kernel.instructions:
            if orig_insn.id in self.insns_to_update:
                new_instructions.append(self.insns_to_update[orig_insn.id])
            else:
                new_instructions.append(orig_insn)
        new_instructions.extend(
            sorted(insns_to_insert.values(), key=lambda insn: insn.id))

        self.updated_iname_to_tag.update(self.kernel.iname_to_tag)
        self.updated_temporary_variables.update(self.kernel.temporary_variables)

        kernel = self.kernel.copy(
            instructions=new_instructions,
            iname_to_tag=self.updated_iname_to_tag,
            temporary_variables=self.updated_temporary_variables)

        from loopy.kernel.tools import assign_automatic_axes
        return assign_automatic_axes(kernel)

    def save(self, temporary, subkernel):
        self.save_or_reload_impl(temporary, subkernel, "save")

    def reload(self, temporary, subkernel):
        self.save_or_reload_impl(temporary, subkernel, "reload")

    def augment_domain_for_save_or_reload(self,
            domain, promoted_temporary, mode, subkernel):
        """
        Add new axes to the domain corresponding to the dimensions of
        `promoted_temporary`. These axes will be used in the save/
        reload stage.
        """
        assert mode in ("save", "reload")
        import islpy as isl

        orig_temporary = promoted_temporary.orig_temporary
        orig_dim = domain.dim(isl.dim_type.set)

        # Tags for newly added inames
        iname_to_tag = {}

        # FIXME: Restrict size of new inames to access footprint.

        # Add dimension-dependent inames.
        dim_inames = []
        domain = domain.add(isl.dim_type.set, len(promoted_temporary.non_hw_dims))

        for dim_idx, dim_size in enumerate(promoted_temporary.non_hw_dims):
            new_iname = self.insn_name_gen("{name}_{mode}_axis_{dim}_{sk}".
                format(name=orig_temporary.name,
                       mode=mode,
                       dim=dim_idx,
                       sk=subkernel))
            domain = domain.set_dim_name(
                isl.dim_type.set, orig_dim + dim_idx, new_iname)

            if orig_temporary.is_local:
                # If the temporary has local scope, then loads / stores can
                # be done in parallel.
                from loopy.kernel.data import AutoFitLocalIndexTag
                iname_to_tag[new_iname] = AutoFitLocalIndexTag()

            dim_inames.append(new_iname)

            # Add size information.
            aff = isl.affs_from_space(domain.space)
            domain &= aff[0].le_set(aff[new_iname])
            from loopy.symbolic import aff_from_expr
            domain &= aff[new_iname].lt_set(aff_from_expr(domain.space, dim_size))

        # FIXME: Use promoted_temporary.hw_inames
        hw_inames = []

        # Add hardware inames duplicates.
        for t_idx, hw_iname in enumerate(promoted_temporary.hw_inames):
            new_iname = self.insn_name_gen("{name}_{mode}_hw_dim_{dim}_{sk}".
                format(name=orig_temporary.name,
                       mode=mode,
                       dim=t_idx,
                       sk=subkernel))
            hw_inames.append(new_iname)
            iname_to_tag[new_iname] = self.kernel.iname_to_tag[hw_iname]

        from loopy.isl_helpers import duplicate_axes
        domain = duplicate_axes(
            domain, promoted_temporary.hw_inames, hw_inames)

        # The operations on the domain above return a Set object, but the
        # underlying domain should be expressible as a single BasicSet.
        domain_list = domain.get_basic_set_list()
        assert domain_list.n_basic_set() == 1
        domain = domain_list.get_basic_set(0)
        return domain, hw_inames, dim_inames, iname_to_tag

# }}}


# {{{ auto save and reload across kernel calls

def save_and_reload_temporaries(knl):
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
    liveness = LivenessAnalysis(knl)
    saver = TemporarySaver(knl)

    insn_query = InstructionQuery(knl)

    for sched_idx, sched_item in enumerate(knl.schedule):

        if isinstance(sched_item, CallKernel):
            # Any written temporary that is live-out needs to be read into
            # memory because of the potential for partial writes.
            if sched_idx == 0:
                # Kernel entry: nothing live
                interesting_temporaries = set()
            else:
                interesting_temporaries = (
                    insn_query.temporaries_read_or_written_in_subkernel(
                        sched_item.kernel_name))

            for temporary in liveness[sched_idx].live_out & interesting_temporaries:
                logger.info("reloading {0} at entry of {1}"
                        .format(temporary, sched_item.kernel_name))
                saver.reload(temporary, sched_item.kernel_name)

        elif isinstance(sched_item, ReturnFromKernel):
            if sched_idx == len(knl.schedule) - 1:
                # Kernel exit: nothing live
                interesting_temporaries = set()
            else:
                interesting_temporaries = (
                    insn_query.temporaries_written_in_subkernel(
                        sched_item.kernel_name))

            for temporary in liveness[sched_idx].live_in & interesting_temporaries:
                logger.info("saving {0} before return of {1}"
                        .format(temporary, sched_item.kernel_name))
                saver.save(temporary, sched_item.kernel_name)

    return saver.finish()

# }}}


# vim: foldmethod=marker

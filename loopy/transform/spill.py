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


import six

from loopy.diagnostic import LoopyError
import loopy as lp

from loopy.kernel.data import auto
from loopy.kernel.instruction import BarrierInstruction
from pytools import memoize_method, Record
from loopy.schedule import (
            EnterLoop, LeaveLoop, RunInstruction,
            CallKernel, ReturnFromKernel, Barrier)


import logging
logger = logging.getLogger(__name__)


# {{{ instruction query utility

class InstructionQuery(object):

    def __init__(self, kernel):
        self.kernel = kernel
        block_bounds = get_block_boundaries(kernel.schedule)
        subkernel_slices = {}
        from six import iteritems
        for start, end in iteritems(block_bounds):
            sched_item = kernel.schedule[start]
            if isinstance(sched_item, CallKernel):
                subkernel_slices[sched_item.kernel_name] = slice(start, end + 1)
        self.subkernel_slices = subkernel_slices

    @memoize_method
    def subkernel_order(self):
        pass

    @memoize_method
    def insns_reading_or_writing(self, var):
        return frozenset(insn.id for insn in self.kernel.instructions
            if var in insn.read_dependency_names()
                or var in insn.assignee_var_names())

    @memoize_method
    def insns_in_subkernel(self, subkernel):
        return frozenset(sched_item.insn_id for sched_item
            in self.kernel.schedule[self.subkernel_slices[subkernel]]
            if isinstance(sched_item, RunInstruction))

    @memoize_method
    def inames_in_subkernel(self, subkernel):
        return frozenset(self.kernel.schedule[self.subkernel_slices[subkernel].start].extra_inames)

    @memoize_method
    def hw_inames(self, insn_id):
        """
        Return the inames that insn runs in and that are tagged as hardware
        parallel.
        """
        from loopy.kernel.data import HardwareParallelTag
        return set(iname for iname in self.kernel.insn_inames(insn_id)
                   if isinstance(self.kernel.iname_to_tag.get(iname), HardwareParallelTag))

    @memoize_method
    def common_hw_inames(self, insn_ids):
        """
        Return the common set of hardware parallel tagged inames among
        the list of instructions.
        """
        # Get the list of hardware inames in which the temporary is defined.
        if len(insn_ids) == 0:
            return set()
        return set.intersection(*(self.hw_inames(id) for id in insn_ids))

# }}}


def get_block_boundaries(schedule):
    """
    Return a dictionary mapping indices of
    :class:`loopy.schedule.BlockBeginItem`s to
    :class:`loopy.schedule.BlockEndItem`s and vice versa.
    """
    from loopy.schedule import (BeginBlockItem, EndBlockItem)
    block_bounds = {}
    active_blocks = []
    for idx, sched_item in enumerate(schedule):
        if isinstance(sched_item, BeginBlockItem):
            active_blocks.append(idx)
        elif isinstance(sched_item, EndBlockItem):
            start = active_blocks.pop()
            block_bounds[start] = idx
            block_bounds[idx] = start
    return block_bounds


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
                    # footprint later on to determine how much to reload/spill.
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


# {{{ spill and reload implementation

class Spiller(object):

    class PromotedTemporary(Record):
        """
        .. attribute:: name

            The name of the new temporary.

        .. attribute:: orig_temporary

            The original temporary variable object.

        .. attribute:: hw_inames

            The common list of hw axes that define the original object.

        .. attribute:: shape_prefix

            A list of expressions, to be added in front of the shape
            of the promoted temporary value
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
            return self.shape_prefix + self.orig_temporary.shape

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
        # i.e. the "extra_args" field of CallKernel
        self.updated_extra_args = {}

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
        shape_prefix = []

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
            shape_prefix.append(
                aff_to_expr(
                    static_max_of_pw_aff(
                        self.kernel.get_iname_bounds(iname).size, False)))

        backing_temporary = self.PromotedTemporary(
            name=self.var_name_gen(temporary.name + ".spill_slot"),
            orig_temporary=temporary,
            shape_prefix=tuple(shape_prefix),
            hw_inames=backing_hw_inames)

        return backing_temporary

    def spill_or_reload_impl(self, temporary, subkernel, mode,
                             promoted_temporary=lp.auto):
        assert mode in ("spill", "reload")

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
            self.augment_domain_for_spill_or_reload(
                dchg.domain, promoted_temporary, mode)

        self.kernel = dchg.get_kernel_with(domain)

        spill_or_load_insn_id = self.insn_name_gen(
            "{name}.{mode}".format(name=temporary, mode=mode))

        def subscript_or_var(agg, subscript=()):
            from pymbolic.primitives import Subscript, Variable
            if len(subscript) == 0:
                return Variable(agg)
            else:
                return Subscript(
                    Variable(agg),
                    tuple(map(Variable, subscript)))

        args = (
            subscript_or_var(
                temporary, dim_inames),
            subscript_or_var(
                promoted_temporary.name, hw_inames + dim_inames))

        if subkernel in self.updated_extra_args:
            self.updated_extra_args[subkernel].append(promoted_temporary.name)
        else:
            self.updated_extra_args[subkernel] = [promoted_temporary.name]

        if mode == "spill":
            args = reversed(args)

        accessing_insns_in_subkernel = (
            self.insn_query.insns_reading_or_writing(temporary) &
            self.insn_query.insns_in_subkernel(subkernel))

        if mode == "spill":
            depends_on = accessing_insns_in_subkernel
            update_deps = frozenset()
        elif mode == "reload":
            depends_on = frozenset()
            update_deps = accessing_insns_in_subkernel

        # Create the load / store instruction.
        from loopy.kernel.data import Assignment
        spill_or_load_insn = Assignment(
            *args,
            id=spill_or_load_insn_id,
            within_inames=self.insn_query.inames_in_subkernel(subkernel) |
                frozenset(hw_inames + dim_inames),
            within_inames_is_final=True,
            depends_on=depends_on,
            boostable=False,
            boostable_into=frozenset())

        self.insns_to_insert.append(spill_or_load_insn)

        for insn_id in update_deps:
            insn = self.insns_to_update.get(insn_id, self.kernel.id_to_insn[insn_id])
            self.insns_to_update[insn_id] = insn.copy(
                depends_on=insn.depends_on | frozenset([spill_or_load_insn_id]))

        self.updated_temporary_variables[promoted_temporary.name] = \
            promoted_temporary.as_variable()

        self.updated_iname_to_tag.update(iname_to_tag)

    @memoize_method
    def finish(self):
        new_instructions = []

        for orig_insn in self.kernel.instructions:
            if orig_insn.id in self.insns_to_update:
                new_instructions.append(self.insns_to_update[orig_insn.id])
            else:
                new_instructions.append(orig_insn)
        new_instructions.extend(self.insns_to_insert)

        new_schedule = []
        for sched_item in self.kernel.schedule:
            if (isinstance(sched_item, CallKernel) and
                    sched_item.kernel_name in self.updated_extra_args):
                new_schedule.append(
                    sched_item.copy(extra_args=(
                        sched_item.extra_args
                        + self.updated_extra_args[sched_item.kernel_name])))
            else:
                new_schedule.append(sched_item)

        self.updated_iname_to_tag.update(self.kernel.iname_to_tag)
        self.updated_temporary_variables.update(self.kernel.temporary_variables)

        return self.kernel.copy(
            schedule=new_schedule,
            instructions=new_instructions,
            iname_to_tag=self.updated_iname_to_tag,
            temporary_variables=self.updated_temporary_variables)

    def spill(self, temporary, subkernel):
        self.spill_or_reload_impl(temporary, subkernel, "spill")

    def reload(self, temporary, subkernel):
        self.spill_or_reload_impl(temporary, subkernel, "reload")

    def get_access_footprint_in_subkernel(self, temporary, subkernel, kind):
        # FIXME: Return some sort of actual non-trivial access footprint.
        assert kind in ("read", "write")

    def augment_domain_for_spill_or_reload(self,
            domain, promoted_temporary, mode):
        """
        Add new axes to the domain corresponding to the dimensions of
        `promoted_temporary`. These axes will be used in the spill/
        reload stage.
        """
        assert mode in ("spill", "reload")
        import islpy as isl

        orig_temporary = promoted_temporary.orig_temporary
        orig_dim = domain.dim(isl.dim_type.set)
        dims_to_insert = len(orig_temporary.shape)

        # Tags for newly added inames
        iname_to_tag = {}

        # Add dimension-dependent inames.
        dim_inames = []
        domain = domain.add(isl.dim_type.set, dims_to_insert)
        for t_idx in range(len(orig_temporary.shape)):
            new_iname = self.insn_name_gen("{name}_{mode}_axis_{dim}".
                format(name=orig_temporary.name,
                       mode=mode,
                       dim=t_idx))
            domain = domain.set_dim_name(
                isl.dim_type.set, orig_dim + t_idx, new_iname)
            if orig_temporary.is_local:
                # If the temporary has local scope, then loads / stores can
                # be done in parallel.
                #from loopy.kernel.data import AutoFitLocalIndexTag
                #iname_to_tag[new_iname] = AutoFitLocalIndexTag()
                pass

            dim_inames.append(new_iname)

            # Add size information.
            aff = isl.affs_from_space(domain.space)
            domain &= aff[0].le_set(aff[new_iname])
            size = orig_temporary.shape[t_idx]
            from loopy.symbolic import aff_from_expr
            domain &= aff[new_iname].lt_set(aff_from_expr(domain.space, size))

        hw_inames = []

        # Add hardware inames duplicates.
        for t_idx, hw_iname in enumerate(promoted_temporary.hw_inames):
            new_iname = self.insn_name_gen("{name}_{mode}_hw_dim_{dim}".
                format(name=orig_temporary.name,
                       mode=mode,
                       dim=t_idx))
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


# {{{ auto spill and reload across kernel calls

"""
TODO:
- flake8ify
- add TODO comments
- document
- assert kernel is scheduled etc
- write a bunch of tests
"""

def spill_and_reload(knl, **kwargs):
    """
    Add instructions to spill and reload temporary variables that are live
    across kernel calls.

    The basic code transformation turns schedule segments:

        t = <...>
        <return followed by call>
        <...> = t

    into this code:

        t = <...>
        t.spill_slot = t
        <return followed by call>
        t = t.spill_slot
        <...> = t

    where `t.spill_slot` is a newly-created global temporary variable.

    :arg knl:
    :arg barriers:
    :returns:
    """
    liveness = LivenessAnalysis(knl)
    spiller = Spiller(knl)

    liveness.print_liveness()

    for sched_idx, sched_item in enumerate(knl.schedule):
        # TODO: Rematerialization
        if isinstance(sched_item, ReturnFromKernel):
            for temporary in liveness[sched_idx].live_in:
                logger.info("spilling {0} before return of {1}"
                        .format(temporary, sched_item.kernel_name))
                spiller.spill(temporary, sched_item.kernel_name)

        elif isinstance(sched_item, CallKernel):
            for temporary in liveness[sched_idx].live_out:
                logger.info("reloading {0} at entry of {1}"
                        .format(temporary, sched_item.kernel_name))
                spiller.reload(temporary, sched_item.kernel_name)

    return spiller.finish()

# }}}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import loopy as lp
    knl = lp.make_kernel(
            "{ [i,j]: 0<=i,j<10 }",
            """
            for i
                <> t_private[i] = 1 {id=define_t_private}
                <> j_ = 1 {id=definej}
                for j
                    ... gbarrier {id=bar}
                    out[j] = j_ {id=setout,dep=bar}
                    ... gbarrier {id=barx,dep=define_t_private,dep=setout}
                    j_ = 10 {id=j1,dep=barx}
                end
                ... gbarrier {id=meow,dep=barx}
                out[i] = t_private[i] {dep=meow}
            end
            """)

    #knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")
    knl = lp.get_one_scheduled_kernel(lp.preprocess_kernel(knl))

    print("SCHEDULED INITIALLY", knl)

    knl = spill_and_reload(knl)

    knl = lp.get_one_scheduled_kernel(knl)
    print(knl)

# vim: foldmethod=marker

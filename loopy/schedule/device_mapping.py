from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2016 Matt Wala"

__license__ = """
(unclear)
"""

# TODO: Matt, please replace the license header
# TODO: Should move to loopy.schedule.device_mapping

from pytools import Record
from loopy.diagnostic import LoopyError


def postprocess(kernel, global_barrier_splitting=False):
    # Analyze the kernel to determine if temporaries are used in a sane way.
    # TODO: Should probably be done in a pre codegen check.
    check_temporary_sanity(kernel)

    if not global_barrier_splitting:
        from loopy.schedule import CallKernel
        new_schedule = (
            [CallKernel(kernel_name=kernel.name,
                        extra_inames=[],
                        extra_temporaries=[])] +
            kernel.schedule +
            [ReturnFromKernel(kernel_name=kernel.name)])
        return kernel.copy(schedule=new_schedule)
    # Split the schedule onto host or device.
    kernel = map_schedule_onto_host_or_device(kernel)
    # Compute which temporaries and inames go into which kernel
    kernel = save_and_restore_temporaries(kernel)
    return kernel


def get_block_boundaries(schedule):
    from loopy.schedule import (
        EnterLoop, LeaveLoop, CallKernel, ReturnFromKernel)
    block_bounds = {}
    active_blocks = []
    for idx, sched_item in enumerate(schedule):
        if isinstance(sched_item, (EnterLoop, CallKernel)):
            active_blocks.append(idx)
        elif isinstance(sched_item, (LeaveLoop, ReturnFromKernel)):
            start = active_blocks.pop()
            block_bounds[start] = idx
            block_bounds[idx] = start
    return block_bounds


def get_common_hw_inames(kernel, insn_ids):
    # Get the list of hardware inames in which the temporary is defined.
    if len(insn_ids) == 0:
        return set()
    id_to_insn = kernel.id_to_insn
    from six.moves import reduce
    return reduce(
        set.intersection,
        (get_hw_inames(kernel, id_to_insn[id]) for id in insn_ids))


# {{{ Use / def analysis

def filter_out_subscripts(exprs):
    result = set()
    from pymbolic.primitives import Subscript
    for expr in exprs:
        if isinstance(expr, Subscript):
            expr = expr.aggregate
        result.add(expr)
    return result


def filter_temporaries(kernel, items):
    from pymbolic.primitives import Subscript, Variable
    result = set()
    for item in items:
        base = item
        if isinstance(base, Subscript):
            base = base.aggregate
        if isinstance(base, Variable):
            base = base.name
        if base in kernel.temporary_variables:
            result.add(item)
    return result


def get_use_set(insn, include_subscripts=True):
    result = insn.read_dependency_names()
    if not include_subscripts:
        result = filter_out_subscripts(result)
    return result


def get_def_set(insn, include_subscripts=True):
    result = insn.write_dependency_names()
    if not include_subscripts:
        result = filter_out_subscripts(result)
    return result


def get_def_and_use_lists_for_all_temporaries(kernel):
    def_lists = dict((t, []) for t in kernel.temporary_variables)
    use_lists = dict((t, []) for t in kernel.temporary_variables)

    # {{{ Gather use-def information

    for insn in kernel.instructions:
        assignees = get_def_set(insn, include_subscripts=False)
        dependencies = get_use_set(insn, include_subscripts=False)

        from pymbolic.primitives import Variable

        for assignee in assignees:
            if isinstance(assignee, Variable):
                assignee = assignee.name
            if assignee in kernel.temporary_variables:
                def_lists[assignee].append(insn.id)

        for dep in dependencies:
            if isinstance(dep, Variable):
                dep = dep.name
            if dep in kernel.temporary_variables:
                use_lists[dep].append(insn.id)

    # }}}

    return def_lists, use_lists

# }}}


def compute_live_temporaries(kernel, schedule):
    live_in = [set() for i in range(len(schedule) + 1)]
    live_out = [set() for i in range(len(schedule))]

    id_to_insn = kernel.id_to_insn
    block_bounds = get_block_boundaries(schedule)

    # {{{ Liveness analysis

    from loopy.schedule import (
        EnterLoop, LeaveLoop, CallKernel, ReturnFromKernel, Barrier, RunInstruction)


    def compute_subrange_liveness(start_idx, end_idx):
        idx = end_idx

        while start_idx <= idx:
            sched_item = schedule[idx]
            if isinstance(sched_item, LeaveLoop):
                start = block_bounds[idx]
                live_in[idx] = live_out[idx] = live_in[idx + 1]
                compute_subrange_liveness(start + 1, idx - 1)
                prev_live_in = live_in[start].copy()
                live_in[start] = live_out[start] = live_in[start + 1]
                # Propagate live values through the loop.
                if live_in[start] != prev_live_in:
                    live_out[idx] |= live_in[start]
                    live_in[idx] = live_out[idx]
                    compute_subrange_liveness(start + 1, idx - 1)
                idx = start - 1

            elif isinstance(sched_item, ReturnFromKernel):
                start = block_bounds[idx]
                live_in[idx] = live_out[idx] = live_in[idx + 1]
                compute_subrange_liveness(start + 1, idx - 1)
                live_in[start] = live_out[start] = live_in[start + 1]
                idx = start - 1

            elif isinstance(sched_item, RunInstruction):
                live_out[idx] = live_in[idx + 1]
                insn = id_to_insn[sched_item.insn_id]
                # `defs` includes subscripts in liveness calculations, so that
                # for code such as the following
                #
                #     Loop i
                #       temp[i] := ...
                #       ...     := f(temp[i])
                #     End Loop
                #
                # the value temp[i] is not marked as live across the loop.
                defs = filter_temporaries(kernel, get_def_set(insn))
                uses = filter_temporaries(kernel, get_use_set(insn))
                live_in[idx] = (live_out[idx] - defs) | uses
                idx -= 1

            elif isinstance(sched_item, Barrier):
                live_in[idx] = live_out[idx] = live_in[idx + 1]
                idx -= 1

            else:
                raise ValueError()

    # }}}

    # Compute live variables
    compute_subrange_liveness(0, len(schedule) - 1)
    live_in = live_in[:-1]

    if 0:
        print("Live-in values:")
        for i, li in enumerate(live_in):
            print("{}: {}".format(i, ", ".join(li)))
        print("Live-out values:")
        for i, lo in enumerate(live_out):
            print("{}: {}".format(i, ", ".join(lo)))

    # Strip off subscripts.
    live_in = [filter_out_subscripts(li) for li in live_in]
    live_out = [filter_out_subscripts(lo) for lo in live_out]

    return live_in, live_out


def save_and_restore_temporaries(kernel):
    # Compute live temporaries.
    live_in, live_out = compute_live_temporaries(kernel, kernel.schedule)

    # Create kernel variables based on live temporaries.
    inter_kernel_temporaries = set()
    from loopy.schedule import CallKernel, ReturnFromKernel, RunInstruction

    for idx, sched_item in enumerate(kernel.schedule):
        if isinstance(sched_item, CallKernel):
            inter_kernel_temporaries |= filter_out_subscripts(live_in[idx])

    def_lists, use_lists = get_def_and_use_lists_for_all_temporaries(kernel)

    # {{{ Determine which temporaries need passing around.

    new_temporaries = {}
    name_gen = kernel.get_var_name_generator()

    from pytools import Record

    class PromotedTemporary(Record):
        """
        .. attribute:: name
        .. attribute:: orig_temporary
        .. attribute:: hw_inames
        .. attribute:: shape_prefix
        """

        def as_variable(self):
            from loopy.kernel.data import TemporaryVariable
            temporary = self.orig_temporary
            return TemporaryVariable(
                name=self.name,
                dtype=temporary.dtype,
                shape=self.new_shape)

        @property
        def new_shape(self):
            return self.shape_prefix + self.orig_temporary.shape


    for temporary in inter_kernel_temporaries:
        from loopy.kernel.data import LocalIndexTag, temp_var_scope

        temporary = kernel.temporary_variables[temporary]
        if temporary.scope == temp_var_scope.GLOBAL:
            # Nothing to be done for global temporaries (I hope)
            continue

        assert temporary.base_storage is None, \
            "Cannot promote temporaries with base_storage to global"

        hw_inames = get_common_hw_inames(kernel, def_lists[temporary.name])
=======
class HostInvokeKernel(Record):
    # TOOD: Should have docstring indicating what attributes can occur
    pass


def map_schedule_onto_host_or_device(kernel):
    from functools import partial
    kernel_name_gen = partial(
            kernel.get_var_name_generator(),
            kernel.name + kernel.target.device_program_name_suffix)
>>>>>>> 55ba3a29f0cf120ae0f74be5d0ee2bb7773ba8e5

        # This takes advantage of the fact that g < l in the alphabet :)
        hw_inames = sorted(hw_inames,
            key=lambda iname: str(kernel.iname_to_tag[iname]))

        shape_prefix = []
        idx = 0
        for iname in hw_inames:
            tag = kernel.iname_to_tag[iname]
            is_local_iname = isinstance(tag, LocalIndexTag)
            if is_local_iname and temporary.scope == temp_var_scope.LOCAL:
                # Restrict shape to that of group inames for locals.
                continue
            from loopy.isl_helpers import static_max_of_pw_aff
            from loopy.symbolic import aff_to_expr
            shape_prefix.append(
                aff_to_expr(
                    static_max_of_pw_aff(
                        kernel.get_iname_bounds(iname).size, False)))

        from loopy.kernel.data import TemporaryVariable
        backing_temporary = PromotedTemporary(
            name=name_gen(temporary.name),
            orig_temporary=temporary,
            shape_prefix=tuple(shape_prefix),
            hw_inames=hw_inames)
        new_temporaries[temporary.name] = backing_temporary

    # }}}

    # {{{ Insert loads and spills of new temporaries.

    new_schedule = []
    new_instructions = []
    new_iname_to_tag = {}

    idx = 0
    schedule = kernel.schedule
    while idx < len(schedule):
        sched_item = schedule[idx]

        if not isinstance(sched_item, CallKernel):
            new_schedule.append(sched_item)
            idx += 1
            continue

        subkernel_defs = set()
        subkernel_uses = set()
        subkernel_prolog = []
        subkernel_epilog = []
        subkernel_schedule = []

        start_idx = idx

        idx += 1
        # Analyze the variables used inside the subkernel.
        while not isinstance(schedule[idx], ReturnFromKernel):
            subkernel_item = schedule[idx]
            subkernel_schedule.append(subkernel_item)
            if isinstance(subkernel_item, RunInstruction):
                insn = kernel.id_to_insn[subkernel_item.insn_id]
                subkernel_defs.update(
                    filter_temporaries(
                        kernel, get_def_set(insn)))
                subkernel_uses.update(
                    filter_temporaries(
                        kernel, get_use_set(insn)))
            idx += 1

        items_to_spill = subkernel_defs & live_out[idx]
        # Need to load items_to_spill, to avoid overwriting entries that the
        # code doesn't touch when doing the spill.
        items_to_load = (subkernel_uses | items_to_spill) & live_in[start_idx]

        # Add arguments.
        new_schedule.append(
            sched_item.copy(extra_args=sorted(
                set(new_temporaries[item].name
                    for item in items_to_spill | items_to_load))))

        from loopy.kernel.tools import DomainChanger
        dchg = DomainChanger(kernel, frozenset(sched_item.extra_inames))
        domain = dchg.get_original_domain()

        import islpy as isl

        # {{{ Add all the loads and spills.

        def augment_domain(item, domain, mode_str):
            temporary = new_temporaries[item]
            orig_size = domain.dim(isl.dim_type.set)
            dims_to_insert = len(temporary.orig_temporary.shape)
            # Add dimension-dependent inames.
            dim_inames = []

            domain = domain.add(isl.dim_type.set, dims_to_insert)
            for t_idx in range(len(temporary.orig_temporary.shape)):
                new_iname = name_gen("{name}.{mode}.dim_{dim}".
                    format(name=temporary.orig_temporary.name,
                           mode=mode_str,
                           dim=orig_size + t_idx))
                domain = domain.set_dim_name(
                    isl.dim_type.set, orig_size + t_idx, new_iname)
                from loopy.kernel.data import auto
                new_iname_to_tag[new_iname] = auto
                dim_inames.append(new_iname)
                # Add size information.
                aff = isl.affs_from_space(domain.space)
                domain &= aff[0].lt_set(aff[iname])
                size = temporary.orig_temporary.shape[t_idx]
                from loopy.symbolic import aff_from_expr
                domain &= aff[iname].lt_set(aff_from_expr(domain.space, size))

            hw_inames = []

            # Add hardware inames duplicates.
            for t_idx, hw_iname in enumerate(temporary.hw_inames):
                new_iname = name_gen("{name}.{mode}.hw_dim_{dim}".
                    format(name=temporary.orig_temporary.name,
                           mode=mode_str,
                           dim=t_idx))
                hw_inames.append(new_iname)
                new_iname_to_tag[new_iname] = kernel.iname_to_tag[hw_iname]

            from loopy.isl_helpers import duplicate_axes
            domain = duplicate_axes(
                domain, temporary.hw_inames, hw_inames)

            # The operations on the domain above return a Set object, but the
            # underlying domain should be expressible as a single BasicSet.
            domain_list = domain.get_basic_set_list()
            assert domain_list.n_basic_set() == 1
            domain = domain_list.get_basic_set(0)
            return domain, hw_inames, dim_inames

        def subscript_or_var(agg, subscript):
            from pymbolic.primitives import Subscript, Variable
            if len(subscript) == 0:
                return Variable(agg)
            else:
                return Subscript(
                    Variable(agg),
                    tuple(map(Variable, subscript)))

        from loopy.kernel.data import Assignment
        for item in items_to_load:
            domain, hw_inames, dim_inames = augment_domain(item, domain, "load")

            # Add a load instruction.
            insn_id = name_gen("{name}.load".format(name=item))

            new_insn = Assignment(
                subscript_or_var(
                    item, dim_inames),
                subscript_or_var(
                    new_temporaries[item].name, hw_inames + dim_inames),
                id=insn_id)

            new_instructions.append(new_insn)
            subkernel_prolog.append(RunInstruction(insn_id=insn_id))

        for item in items_to_spill:
            domain, hw_inames, dim_inames = augment_domain(item, domain, "spill")

            # Add a load instruction.
            insn_id = name_gen("{name}.spill".format(name=item))

            new_insn = Assignment(
                subscript_or_var(
                    new_temporaries[item].name, hw_inames + dim_inames),
                subscript_or_var(
                    item, dim_inames),
                id=insn_id)

            new_instructions.append(new_insn)
            subkernel_epilog.append(RunInstruction(insn_id=insn_id))

        # }}}

        # DomainChanger returns a new kernel object, so we need to replace the
        # kernel here.
        kernel = dchg.get_kernel_with(domain)

        new_schedule.extend(
            subkernel_prolog +
            subkernel_schedule +
            subkernel_epilog)

        # ReturnFromKernel
        new_schedule.append(schedule[idx])
        idx += 1

    # }}}

    new_iname_to_tag.update(kernel.iname_to_tag)
    new_temporary_variables = dict(
        (t.name, t.as_variable()) for t in new_temporaries.values())
    new_temporary_variables.update(kernel.temporary_variables)

    kernel = kernel.copy(
        iname_to_tag=new_iname_to_tag,
        temporary_variables=new_temporary_variables,
        instructions=kernel.instructions + new_instructions,
        schedule=new_schedule
        )

    return kernel


def map_schedule_onto_host_or_device(kernel):
    from loopy.schedule import (
        RunInstruction, EnterLoop, LeaveLoop, Barrier,
        CallKernel, ReturnFromKernel)

    # TODO: Assert that the kernel has been scheduled, etc.
    schedule = kernel.schedule

    loop_bounds = get_block_boundaries(schedule)

    # {{{ Inner mapper function

    dummy_call = CallKernel(kernel_name="", extra_args=[], extra_inames=[])
    dummy_return = ReturnFromKernel(kernel_name="")

    def inner_mapper(start_idx, end_idx, new_schedule):
        schedule_required_splitting = False

        i = start_idx
        current_chunk = []
        while i <= end_idx:
            sched_item = schedule[i]

            if isinstance(sched_item, RunInstruction):
                current_chunk.append(sched_item)
                i += 1

            elif isinstance(sched_item, EnterLoop):
                loop_end = loop_bounds[i]
                inner_schedule = []
                loop_required_splitting = inner_mapper(
                    i + 1, loop_end - 1, inner_schedule)

                start_item = schedule[i]
                end_item = schedule[loop_end]

                i = loop_end + 1

                if loop_required_splitting:
                    schedule_required_splitting = True
                    if current_chunk:
                        new_schedule.extend(
                            [dummy_call.copy()] +
                            current_chunk +
                            [dummy_return.copy()])
                    new_schedule.extend(
                        [start_item] +
                        inner_schedule +
                        [end_item])
                    current_chunk = []
                else:
                    current_chunk.extend(
                        [start_item] +
                        inner_schedule +
                        [end_item])

            elif isinstance(sched_item, Barrier):
                if sched_item.kind == "global":
                    # Wrap the current chunk into a kernel call.
                    schedule_required_splitting = True
                    if current_chunk:
                        new_schedule.extend(
                            [dummy_call.copy()] +
                            current_chunk +
                            [dummy_return.copy()])
                    current_chunk = []
                else:
                    current_chunk.append(sched_item)
                i += 1
            else:
                raise LoopyError("unexepcted type of schedule item: %s"
                        % type(sched_item).__name__)

        if current_chunk and schedule_required_splitting:
            # Wrap remainder of schedule into a kernel call.
            new_schedule.extend(
                [dummy_call.copy()] +
                current_chunk +
                [dummy_return.copy()])
        else:
            new_schedule.extend(current_chunk)

        return schedule_required_splitting

    # }}}

    new_schedule = []
    split_kernel = inner_mapper(0, len(schedule) - 1, new_schedule)
    if not split_kernel:
        # Wrap everything into a kernel call.
        new_schedule = (
            [dummy_call.copy()] +
            new_schedule +
            [dummy_return.copy()])

    # Assign names, inames to CallKernel / ReturnFromKernel instructions
    inames = []
    from pytools import UniqueNameGenerator
    kernel_name_gen = UniqueNameGenerator(forced_prefix=kernel.name)
    for idx, sched_item in enumerate(new_schedule):
        if isinstance(sched_item, CallKernel):
            last_kernel_name = kernel_name_gen()
            new_schedule[idx] = sched_item.copy(
                kernel_name=last_kernel_name,
                extra_inames=list(inames))
        elif isinstance(sched_item, ReturnFromKernel):
            new_schedule[idx] = sched_item.copy(
                kernel_name=last_kernel_name)
        elif isinstance(sched_item, EnterLoop):
            inames.append(sched_item.iname)
        elif isinstance(sched_item, LeaveLoop):
            inames.pop()

    new_kernel = kernel.copy(schedule=new_schedule)
    return new_kernel


def get_hw_inames(kernel, insn):
    from loopy.kernel.data import HardwareParallelTag
    return set(iname for iname in kernel.insn_inames(insn)
        if isinstance(kernel.iname_to_tag.get(iname), HardwareParallelTag))


def analyze_temporaries(kernel):
    # {{{ Analyze uses of temporaries by hardware loops

    def_lists, use_lists = get_def_and_use_lists_for_all_temporaries(kernel)

    for temporary in sorted(def_lists):
        def_list = def_lists[temporary]

        # Ensure that no use of the temporary is at a loop nesting level
        # that is "more general" than the definition.
        for use in use_lists[temporary]:
            if not hw_inames <= get_hw_inames(insn):
                raise ValueError(
                    "Temporary variable `{temporary}` gets used in a more "
                    "general hardware parallel loop than it is defined. "
                    "(used by instruction id `{id}`, inames: {use_inames}) "
                    "(defined in inames: {def_inames}).".format(
                        temporary=temporary,
                        id=use.id,
                        use_inames=", ".join(sorted(get_hw_inames(insn))),
                        def_inames=", ".join(sorted(hw_inames))))

    # }}}

# }}}

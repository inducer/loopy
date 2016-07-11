from __future__ import division, absolute_import, print_function

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
from loopy.kernel.data import TemporaryVariable, temp_var_scope
from loopy.schedule import (Barrier, BeginBlockItem, CallKernel, EndBlockItem,
                            EnterLoop, LeaveLoop, ReturnFromKernel,
                            RunInstruction)
from pytools import Record, memoize_method


def map_schedule_onto_host_or_device(kernel):
    from loopy.kernel import kernel_state
    assert kernel.state == kernel_state.SCHEDULED

    from functools import partial
    device_prog_name_gen = partial(
                kernel.get_var_name_generator(),
                kernel.target.device_program_name_prefix
                + kernel.name
                + kernel.target.device_program_name_suffix)

    if not kernel.target.split_kernel_at_global_barriers():
        new_schedule = (
            [CallKernel(kernel_name=device_prog_name_gen(),
                        extra_args=[],
                        extra_inames=[])] +
            list(kernel.schedule) +
            [ReturnFromKernel(kernel_name=kernel.name)])
        kernel = kernel.copy(schedule=new_schedule)
    else:
        kernel = map_schedule_onto_host_or_device_impl(
                kernel, device_prog_name_gen)

    return restore_and_save_temporaries(
        add_extra_args_to_schedule(kernel))


# {{{ Schedule / instruction utilities

def get_block_boundaries(schedule):
    """
    Return a dictionary mapping indices of
    :class:`loopy.schedule.BlockBeginItem`s to
    :class:`loopy.schedule.BlockEndItem`s and vice versa.
    """
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


def get_hw_inames(kernel, insn):
    """
    Return the inames that insn runs in and that are tagged as hardware
    parallel.
    """
    from loopy.kernel.data import HardwareParallelTag
    return set(iname for iname in kernel.insn_inames(insn)
        if isinstance(kernel.iname_to_tag.get(iname), HardwareParallelTag))


def get_common_hw_inames(kernel, insn_ids):
    """
    Return the common set of hardware parallel tagged inames among
    the list of instructions.
    """
    # Get the list of hardware inames in which the temporary is defined.
    if len(insn_ids) == 0:
        return set()
    id_to_insn = kernel.id_to_insn
    from six.moves import reduce
    return reduce(
        set.intersection,
        (get_hw_inames(kernel, id_to_insn[id]) for id in insn_ids))

# }}}


# {{{ Use / def utilities

def filter_out_subscripts(exprs):
    """
    Remove subscripts from expressions in `exprs`.
    """
    result = set()
    from pymbolic.primitives import Subscript
    for expr in exprs:
        if isinstance(expr, Subscript):
            expr = expr.aggregate
        result.add(expr)
    return result


def filter_items_by_varname(pred, kernel, items):
    """
    Keep only the values in `items` whose variable names satisfy `pred`.
    """
    from pymbolic.primitives import Subscript, Variable
    result = set()
    for item in items:
        base = item
        if isinstance(base, Subscript):
            base = base.aggregate
        if isinstance(base, Variable):
            base = base.name
        if pred(kernel, base):
            result.add(item)
    return result


from functools import partial

filter_temporaries = partial(filter_items_by_varname,
    lambda kernel, name: name in kernel.temporary_variables)

filter_scalar_temporaries = partial(filter_items_by_varname,
    lambda kernel, name: name in kernel.temporary_variables and
        len(kernel.temporary_variables[name].shape) == 0)


def get_use_set(insn, include_subscripts=True):
    """
    Return the use-set of the instruction, for liveness analysis.
    """
    result = insn.read_dependency_names()
    if not include_subscripts:
        result = filter_out_subscripts(result)
    return result


def get_def_set(insn, include_subscripts=True):
    """
    Return the def-set of the instruction, for liveness analysis.
    """
    result = insn.write_dependency_names()
    if not include_subscripts:
        result = filter_out_subscripts(result)
    return result


def get_def_and_use_lists_for_all_temporaries(kernel):
    """
    Return a pair `def_lists`, `use_lists` which map temporary variable
    names to lists of instructions where they are defined or used.
    """
    def_lists = dict((t, []) for t in kernel.temporary_variables)
    use_lists = dict((t, []) for t in kernel.temporary_variables)

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

    return def_lists, use_lists


def get_temporaries_defined_and_used_in_subrange(
        kernel, schedule, start_idx, end_idx):
    defs = set()
    uses = set()

    for idx in range(start_idx, end_idx + 1):
        sched_item = schedule[idx]
        if isinstance(sched_item, RunInstruction):
            insn = kernel.id_to_insn[sched_item.insn_id]
            defs.update(
                filter_temporaries(
                    kernel, get_def_set(insn)))
            uses.update(
                filter_temporaries(
                    kernel, get_use_set(insn)))

    return defs, uses

# }}}


# {{{ Liveness analysis

def compute_live_temporaries(kernel, schedule):
    """
    Compute live-in and live-out sets for temporary variables.
    """
    live_in = [set() for i in range(len(schedule) + 1)]
    live_out = [set() for i in range(len(schedule))]

    id_to_insn = kernel.id_to_insn
    block_bounds = get_block_boundaries(schedule)

    # {{{ Liveness analysis implementation

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
                defs = filter_scalar_temporaries(kernel,
                    get_def_set(insn, include_subscripts=False))
                uses = filter_temporaries(kernel,
                    get_use_set(insn, include_subscripts=False))
                live_in[idx] = (live_out[idx] - defs) | uses
                idx -= 1

            elif isinstance(sched_item, Barrier):
                live_in[idx] = live_out[idx] = live_in[idx + 1]
                idx -= 1
            else:
                raise LoopyError("unexepcted type of schedule item: %s"
                        % type(sched_item).__name__)

    # }}}

    # Compute live variables
    compute_subrange_liveness(0, len(schedule) - 1)
    live_in = live_in[:-1]

    if 0:
        print(kernel)
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

# }}}


# {{{ Temporary promotion

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
        return TemporaryVariable(
            name=self.name,
            dtype=temporary.dtype,
            scope=temp_var_scope.GLOBAL,
            shape=self.new_shape)

    @property
    def new_shape(self):
        return self.shape_prefix + self.orig_temporary.shape


def determine_temporaries_to_promote(kernel, temporaries, name_gen):
    """
    :returns: A :class:`dict` mapping temporary names from `temporaries` to
              :class:`PromotedTemporary` objects
    """
    new_temporaries = {}

    def_lists, use_lists = get_def_and_use_lists_for_all_temporaries(kernel)

    from loopy.kernel.data import LocalIndexTag

    for temporary in temporaries:
        temporary = kernel.temporary_variables[temporary]
        if temporary.scope == temp_var_scope.GLOBAL:
            # Nothing to be done for global temporaries (I hope)
            continue

        assert temporary.base_storage is None, \
            "Cannot promote temporaries with base_storage to global"

        hw_inames = get_common_hw_inames(kernel,
            def_lists[temporary.name] + use_lists[temporary.name])

        # This takes advantage of the fact that g < l in the alphabet :)
        hw_inames = sorted(hw_inames,
            key=lambda iname: str(kernel.iname_to_tag[iname]))

        shape_prefix = []

        backing_hw_inames = []
        for iname in hw_inames:
            tag = kernel.iname_to_tag[iname]
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
                        kernel.get_iname_bounds(iname).size, False)))

        backing_temporary = PromotedTemporary(
            name=name_gen(temporary.name),
            orig_temporary=temporary,
            shape_prefix=tuple(shape_prefix),
            hw_inames=backing_hw_inames)
        new_temporaries[temporary.name] = backing_temporary

    return new_temporaries

# }}}


# {{{ Domain augmentation

def augment_domain_for_temporary_promotion(
        kernel, domain, promoted_temporary, mode, name_gen):
    """
    Add new axes to the domain corresponding to the dimensions of
    `promoted_temporary`.
    """
    import islpy as isl

    orig_temporary = promoted_temporary.orig_temporary
    orig_dim = domain.dim(isl.dim_type.set)
    dims_to_insert = len(orig_temporary.shape)

    iname_to_tag = {}

    # Add dimension-dependent inames.
    dim_inames = []

    domain = domain.add(isl.dim_type.set, dims_to_insert)
    for t_idx in range(len(orig_temporary.shape)):
        new_iname = name_gen("{name}_{mode}_dim_{dim}".
            format(name=orig_temporary.name,
                   mode=mode,
                   dim=orig_dim + t_idx))
        domain = domain.set_dim_name(
            isl.dim_type.set, orig_dim + t_idx, new_iname)
        #from loopy.kernel.data import auto
        #iname_to_tag[new_iname] = auto
        dim_inames.append(new_iname)

        # Add size information.
        aff = isl.affs_from_space(domain.space)
        domain &= aff[0].le_set(aff[new_iname])
        size = orig_temporary.shape[t_idx]
        from loopy.symbolic import aff_from_expr
        domain &= aff[new_iname].le_set(aff_from_expr(domain.space, size))

    hw_inames = []

    # Add hardware inames duplicates.
    for t_idx, hw_iname in enumerate(promoted_temporary.hw_inames):
        new_iname = name_gen("{name}_{mode}_hw_dim_{dim}".
            format(name=orig_temporary.name,
                   mode=mode,
                   dim=t_idx))
        hw_inames.append(new_iname)
        iname_to_tag[new_iname] = kernel.iname_to_tag[hw_iname]

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


def restore_and_save_temporaries(kernel):
    """
    Add code that loads / spills the temporaries in the kernel which are
    live across sub-kernel calls.
    """
    # Compute live temporaries.
    live_in, live_out = compute_live_temporaries(kernel, kernel.schedule)

    # Create kernel variables based on live temporaries.
    inter_kernel_temporaries = set()

    call_count = 0
    for idx, sched_item in enumerate(kernel.schedule):
        if isinstance(sched_item, CallKernel):
            inter_kernel_temporaries |= filter_out_subscripts(live_in[idx])
            call_count += 1

    if call_count == 1:
        # Single call corresponds to a kernel which has not been split -
        # no need for restores / spills of temporaries.
        return kernel

    name_gen = kernel.get_var_name_generator()
    new_temporaries = determine_temporaries_to_promote(
        kernel, inter_kernel_temporaries, name_gen)

    # {{{ Insert loads and spills of new temporaries

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

        subkernel_prolog = []
        subkernel_epilog = []
        subkernel_schedule = []

        start_idx = idx
        idx += 1
        while not isinstance(schedule[idx], ReturnFromKernel):
            subkernel_schedule.append(schedule[idx])
            idx += 1

        subkernel_defs, subkernel_uses = \
            get_temporaries_defined_and_used_in_subrange(
                kernel, schedule, start_idx + 1, idx - 1)

        from loopy.kernel.data import temp_var_scope
        # Filter out temporaries that are global.
        subkernel_globals = set(
            tval for tval in subkernel_defs | subkernel_uses
            if kernel.temporary_variables[tval].scope == temp_var_scope.GLOBAL)

        tvals_to_spill = (subkernel_defs - subkernel_globals) & live_out[idx]
        # Need to load tvals_to_spill, to avoid overwriting entries that the
        # code doesn't touch when doing the spill.
        tvals_to_load = ((subkernel_uses - subkernel_globals)
            | tvals_to_spill) & live_in[start_idx]

        # Add new arguments.
        sched_item = sched_item.copy(
            extra_args=sched_item.extra_args
            + sorted(new_temporaries[tv].name
                     for tv in tvals_to_load | tvals_to_spill))

        # {{{ Add all the loads and spills.

        def insert_loads_or_spills(tvals, mode):
            assert mode in ["load", "spill"]
            local_temporaries = set()

            code_block = \
                subkernel_prolog if mode == "load" else subkernel_epilog

            new_kernel = kernel

            for tval in tvals:
                from loopy.kernel.tools import DomainChanger
                tval_hw_inames = new_temporaries[tval].hw_inames
                dchg = DomainChanger(kernel,
                    frozenset(sched_item.extra_inames + tval_hw_inames))
                domain = dchg.domain

                domain, hw_inames, dim_inames, itt = \
                    augment_domain_for_temporary_promotion(
                        new_kernel, domain, new_temporaries[tval], mode,
                        name_gen)
                new_iname_to_tag.update(itt)

                new_kernel = dchg.get_kernel_with(domain)

                # Add the load / spill instruction.
                insn_id = name_gen("{name}.{mode}".format(name=tval, mode=mode))

                def subscript_or_var(agg, subscript):
                    from pymbolic.primitives import Subscript, Variable
                    if len(subscript) == 0:
                        return Variable(agg)
                    else:
                        return Subscript(
                            Variable(agg),
                            tuple(map(Variable, subscript)))

                args = (
                    subscript_or_var(
                        tval, dim_inames),
                    subscript_or_var(
                        new_temporaries[tval].name, hw_inames + dim_inames))

                if mode == "spill":
                    args = reversed(args)

                from loopy.kernel.data import Assignment
                new_insn = Assignment(*args, id=insn_id)

                new_instructions.append(new_insn)

                loop_begin = [EnterLoop(iname=iname) for iname in dim_inames]
                loop_end = list(reversed([
                    LeaveLoop(iname=iname) for iname in dim_inames]))
                code_block.extend(
                    loop_begin +
                    [RunInstruction(insn_id=insn_id)] +
                    loop_end)
                if new_temporaries[tval].orig_temporary.is_local:
                    local_temporaries.add(new_temporaries[tval].name)

            # After loading / before spilling local temporaries, we need to
            # insert a barrier.
            if local_temporaries:
                if mode == "load":
                    subkernel_prolog.append(
                        Barrier(kind="local",
                                comment="for loads of {0}".format(
                                    ", ".join(sorted(local_temporaries)))))
                else:
                    subkernel_epilog.insert(0,
                        Barrier(kind="local",
                                comment="for spills of {0}".format(
                                    ", ".join(sorted(local_temporaries)))))
            return new_kernel

        kernel = insert_loads_or_spills(tvals_to_load, "load")
        kernel = insert_loads_or_spills(tvals_to_spill, "spill")

        # }}}

        new_schedule.extend(
            [sched_item] +
            subkernel_prolog +
            subkernel_schedule +
            subkernel_epilog +
            # ReturnFromKernel
            [schedule[idx]])

        # ReturnFromKernel
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


def add_extra_args_to_schedule(kernel):
    """
    Fill the `extra_args` fields in all the :class:`loopy.schedule.CallKernel`
    instructions in the schedule with global temporaries.
    """
    new_schedule = []

    block_bounds = get_block_boundaries(kernel.schedule)
    for idx, sched_item in enumerate(kernel.schedule):
        if isinstance(sched_item, CallKernel):
            defs, uses = get_temporaries_defined_and_used_in_subrange(
                   kernel, kernel.schedule, idx + 1, block_bounds[idx] - 1)
            # Filter out temporaries that are global.
            extra_args = (tv for tv in defs | uses if
                kernel.temporary_variables[tv].scope == temp_var_scope.GLOBAL
                and
                kernel.temporary_variables[tv].initializer is None)
            new_schedule.append(sched_item.copy(extra_args=sorted(extra_args)))
        else:
            new_schedule.append(sched_item)

    return kernel.copy(schedule=new_schedule)


def map_schedule_onto_host_or_device_impl(kernel, device_prog_name_gen):
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

    # Assign names, extra_inames to CallKernel / ReturnFromKernel instructions
    inames = []

    for idx, sched_item in enumerate(new_schedule):
        if isinstance(sched_item, CallKernel):
            last_kernel_name = device_prog_name_gen()
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

"""Loop nest build top-level control/hoisting."""


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
from functools import partial

from loopy.codegen.result import merge_codegen_results, wrap_in_if
from loopy.schedule import (
        EnterLoop, LeaveLoop, RunInstruction, Barrier, CallKernel,
        gather_schedule_block, generate_sub_sched_items)
from loopy.diagnostic import LoopyError


def generate_code_for_sched_index(codegen_state, sched_index):
    kernel = codegen_state.kernel
    sched_item = kernel.linearization[sched_index]

    if isinstance(sched_item, CallKernel):
        assert not codegen_state.is_generating_device_code

        from loopy.schedule import (gather_schedule_block, get_insn_ids_for_block_at)
        _, past_end_i = gather_schedule_block(kernel.linearization, sched_index)
        assert past_end_i <= codegen_state.schedule_index_end

        new_codegen_state = codegen_state.copy(
                is_generating_device_code=True,
                gen_program_name=sched_item.kernel_name,
                schedule_index_end=past_end_i-1,
                )

        from loopy.codegen.result import generate_host_or_device_program
        codegen_result = generate_host_or_device_program(
                new_codegen_state, sched_index)

        if codegen_state.is_entrypoint:
            glob_grid, loc_grid = kernel.get_grid_sizes_for_insn_ids_as_exprs(
                    get_insn_ids_for_block_at(kernel.linearization, sched_index),
                    codegen_state.callables_table)
            return merge_codegen_results(codegen_state, [
                codegen_result,

                codegen_state.ast_builder.get_kernel_call(
                    codegen_state,
                    sched_item.kernel_name,
                    glob_grid, loc_grid)
                ])
        else:
            # do not generate host code for non-entrypoint kernels
            return codegen_result

    elif isinstance(sched_item, EnterLoop):
        from loopy.kernel.data import (UnrolledIlpTag, UnrollTag,
                ForceSequentialTag, LoopedIlpTag, VectorizeTag,
                InameImplementationTag, UnrollHintTag,
                InOrderSequentialSequentialTag, filter_iname_tags_by_type)

        tags = kernel.iname_tags_of_type(sched_item.iname, InameImplementationTag)
        tags = tuple(tag for tag in tags if tag)

        from loopy.codegen.loop import (
                generate_unroll_loop,
                generate_vectorize_loop,
                generate_sequential_loop_dim_code)

        if filter_iname_tags_by_type(tags, (UnrollTag, UnrolledIlpTag)):
            func = generate_unroll_loop
        elif filter_iname_tags_by_type(tags, VectorizeTag):
            func = generate_vectorize_loop
        elif filter_iname_tags_by_type(tags, UnrollHintTag):
            unroll_tags = filter_iname_tags_by_type(tags, UnrollHintTag)
            hints = [codegen_state.ast_builder.emit_unroll_hint(tag.value)
                    for tag in unroll_tags]
            func = partial(generate_sequential_loop_dim_code, hints=hints)
        elif not tags or filter_iname_tags_by_type(tags, (LoopedIlpTag,
                    ForceSequentialTag, InOrderSequentialSequentialTag)):
            func = partial(generate_sequential_loop_dim_code, hints=[])
        else:
            raise RuntimeError("encountered (invalid) EnterLoop "
                    "for '%s', tagged '%s'"
                    % (sched_item.iname, ", ".join(str(tag) for tag in tags)))

        return func(codegen_state, sched_index)

    elif isinstance(sched_item, Barrier):
        # {{{ emit barrier code

        from loopy.codegen.result import CodeGenerationResult

        if codegen_state.is_generating_device_code:
            barrier_ast = codegen_state.ast_builder.emit_barrier(
                    sched_item.synchronization_kind, sched_item.mem_kind,
                    sched_item.comment)
            if sched_item.originating_insn_id:
                return CodeGenerationResult.new(
                        codegen_state,
                        sched_item.originating_insn_id,
                        barrier_ast,
                        codegen_state.implemented_domain)
            else:
                return barrier_ast
        else:
            # host code
            if sched_item.synchronization_kind in ["global", "local"]:
                # host code is assumed globally and locally synchronous
                return CodeGenerationResult(
                        host_program=None,
                        device_programs=[],
                        implemented_domains={})

            else:
                raise LoopyError("do not know how to emit code for barrier "
                                 "synchronization kind '%s'" "in host code"
                                 % sched_item.synchronization_kind)

        # }}}

    elif isinstance(sched_item, RunInstruction):
        insn = kernel.id_to_insn[sched_item.insn_id]

        from loopy.codegen.instruction import generate_instruction_code
        return codegen_state.try_vectorized(
                "instruction %s" % insn.id,
                lambda inner_cgs: generate_instruction_code(inner_cgs, insn))

    else:
        raise RuntimeError("unexpected schedule item type: %s"
                % type(sched_item))


def get_required_predicates(kernel, sched_index):
    result = None
    for _, sched_item in generate_sub_sched_items(kernel.linearization, sched_index):
        if isinstance(sched_item, Barrier):
            my_preds = frozenset()
        elif isinstance(sched_item, RunInstruction):
            my_preds = kernel.id_to_insn[sched_item.insn_id].predicates
        else:
            raise RuntimeError("unexpected schedule item type: %s"
                    % type(sched_item))

        if result is None:
            result = my_preds
        else:
            result = result & my_preds

    if result is None:
        result = frozenset()

    return result


def group_by(entry, key, merge):
    if not entry:
        return entry

    result = []
    previous = entry[0]

    for item in entry[1:]:
        if key(previous) == key(item):
            previous = merge(previous, item)

        else:
            result.append(previous)
            previous = item

    result.append(previous)
    return result


def build_loop_nest(codegen_state, schedule_index):
    # Most of the complexity of this function goes towards finding groups of
    # instructions that can be nested inside a shared conditional.

    kernel = codegen_state.kernel

    # If the AST builder does not implement conditionals, we can save us
    # some work about hoisting conditionals and directly go into recursion.
    if not codegen_state.ast_builder.can_implement_conditionals:
        result = []
        inner = generate_code_for_sched_index(codegen_state, schedule_index)
        if inner is not None:
            result.append(inner)
        return merge_codegen_results(codegen_state, result)

    # {{{ pass 1: pre-scan schedule for my schedule item's siblings' indices

    # i.e. go up to the next LeaveLoop, and skip over inner loops.

    my_sched_indices = []

    i = schedule_index
    while i < codegen_state.schedule_index_end:
        sched_item = kernel.linearization[i]

        if isinstance(sched_item, LeaveLoop):
            break

        my_sched_indices.append(i)

        if isinstance(sched_item, (EnterLoop, CallKernel)):
            _, i = gather_schedule_block(kernel.linearization, i)
            assert i <= codegen_state.schedule_index_end, \
                    "schedule block extends beyond schedule_index_end"

        elif isinstance(sched_item, Barrier):
            i += 1

        elif isinstance(sched_item, RunInstruction):
            i += 1
        else:
            raise RuntimeError("unexpected schedule item type: %s"
                    % type(sched_item))

    del i

    # }}}

    # {{{ pass 2: find admissible conditional inames for each sibling schedule item

    from pytools import ImmutableRecord

    class ScheduleIndexInfo(ImmutableRecord):
        """
        .. attribute:: schedule_index
        .. attribute:: admissible_cond_inames
        .. attribute:: required_predicates
        .. attribute:: used_inames_within
        """

    from loopy.schedule import find_used_inames_within
    from loopy.codegen.bounds import get_usable_inames_for_conditional

    sched_index_info_entries = [
            ScheduleIndexInfo(
                schedule_indices=[i],
                admissible_cond_inames=(
                    get_usable_inames_for_conditional(kernel, i,
                        codegen_state.codegen_cachemanager)),
                required_predicates=get_required_predicates(kernel, i),
                used_inames_within=find_used_inames_within(kernel, i)
                )
            for i in my_sched_indices
            ]

    sched_index_info_entries = group_by(
            sched_index_info_entries,
            key=lambda sii: (
                sii.admissible_cond_inames,
                sii.required_predicates,
                sii.used_inames_within),
            merge=lambda sii1, sii2: sii1.copy(
                schedule_indices=(
                    sii1.schedule_indices
                    +
                    sii2.schedule_indices)))

    # }}}

    # {{{ pass 3: greedily group schedule items that share admissible inames

    from pytools import memoize_method

    class BoundsCheckCache:
        def __init__(self, kernel, impl_domain):
            self.kernel = kernel
            self.impl_domain = impl_domain

        @memoize_method
        def __call__(self, check_inames):
            if not check_inames:
                return []

            domain = isl.align_spaces(
                    self.kernel.get_inames_domain(check_inames),
                    self.impl_domain, obj_bigger_ok=True)
            from loopy.codegen.bounds import get_approximate_convex_bounds_checks
            # Each instruction individually gets its bounds checks,
            # so we can safely overapproximate here.
            return get_approximate_convex_bounds_checks(domain,
                    check_inames, self.impl_domain, self.kernel.cache_manager)

    def build_insn_group(sched_index_info_entries, codegen_state,
            done_group_lengths=frozenset()):
        """
        :arg done_group_lengths: A set of group lengths (integers) that grows
            from empty to include the longest found group and downwards with every
            recursive call.  It serves to prevent infinite recursion by preventing
            recursive calls from doing anything about groups that are too small.
        """

        from loopy.symbolic import get_dependencies

        # The rough plan here is that build_insn_group starts out with the
        # entirety of the current schedule item's downward siblings (i.e. all
        # the ones up to the next LeaveLoop). It will then iterate upward to
        # find the largest usable conditional hoist group.
        #
        # It will then call itself recursively, telling its recursive instances
        # to ignore the hoist group it just found by adding that group length
        # to done_group_length. (It'll also chop the set of schedule indices
        # considered down so that a callee cannot find a *longer* hoist group.)
        #
        # Upon return the hoist is wrapped around the returned code and
        # build_insn_group calls itself for the remainder of schedule indices
        # that were not in the hoist group.

        if not sched_index_info_entries:
            return []

        origin_si_entry = sched_index_info_entries[0]
        current_iname_set = origin_si_entry.admissible_cond_inames
        current_pred_set = (origin_si_entry.required_predicates
                - codegen_state.implemented_predicates)

        # {{{ grow schedule item group

        # Keep growing schedule item group as long as group fulfills minimum
        # size requirement.

        bounds_check_cache = BoundsCheckCache(
                kernel, codegen_state.implemented_domain)

        found_hoists = []

        candidate_group_length = 1
        while candidate_group_length <= len(sched_index_info_entries):
            if candidate_group_length in done_group_lengths:
                candidate_group_length += 1
                continue

            current_iname_set = (
                    current_iname_set
                    & sched_index_info_entries[candidate_group_length-1]
                    .admissible_cond_inames)
            current_pred_set = (
                    current_pred_set
                    & sched_index_info_entries[candidate_group_length-1]
                    .required_predicates)

            current_pred_set = frozenset(
                    pred for pred in current_pred_set
                    if get_dependencies(pred) & kernel.all_inames()
                    <= current_iname_set)

            # {{{ see which inames are actually used in group

            # And only generate conditionals for those.
            used_inames = set()
            for sched_index_info_entry in \
                    sched_index_info_entries[0:candidate_group_length]:
                used_inames |= sched_index_info_entry.used_inames_within

            # }}}

            only_unshared_inames = kernel._remove_inames_for_shared_hw_axes(
                    current_iname_set & used_inames)

            bounds_checks = bounds_check_cache(only_unshared_inames)

            if (bounds_checks  # found a bounds check
                    or current_pred_set
                    or candidate_group_length == 1):
                # length-1 must always be an option to reach the recursion base
                # case below
                found_hoists.append((candidate_group_length,
                    bounds_checks, current_pred_set))

            if not bounds_checks and not current_pred_set:
                # already no more checks possible, let's not waste time
                # checking longer groups.
                break

            candidate_group_length += 1

        # }}}

        # pick largest such group
        group_length, bounds_checks, pred_checks = max(found_hoists)

        check_set = None
        for cns in bounds_checks:
            cns_set = (isl.BasicSet.universe(cns.get_space())
                    .add_constraint(cns))

            if check_set is None:
                check_set = cns_set
            else:
                check_set, cns_set = isl.align_two(check_set, cns_set)
                check_set = check_set.intersect(cns_set)

        if check_set is None:
            new_codegen_state = codegen_state
            is_empty = False
        else:
            is_empty = check_set.is_empty()
            new_codegen_state = codegen_state.intersect(check_set)

        if pred_checks:
            new_codegen_state = new_codegen_state.copy(
                    implemented_predicates=new_codegen_state.implemented_predicates
                    | pred_checks)

        if is_empty:
            result = []
        else:
            if group_length == 1:
                # group only contains starting schedule item
                def gen_code(inner_codegen_state):
                    result = []
                    for i in origin_si_entry.schedule_indices:
                        inner = generate_code_for_sched_index(
                            inner_codegen_state, i)

                        if inner is not None:
                            result.append(inner)

                    return result

            else:
                # recurse with a bigger done_group_lengths
                def gen_code(inner_codegen_state):
                    return build_insn_group(
                            sched_index_info_entries[0:group_length],
                            inner_codegen_state,
                            done_group_lengths=(
                                done_group_lengths | {group_length}))

            # gen_code returns a list

            if bounds_checks or pred_checks:
                from loopy.symbolic import constraint_to_cond_expr

                prev_gen_code = gen_code

                def gen_code(inner_codegen_state):  # noqa pylint:disable=function-redefined
                    condition_exprs = [
                            constraint_to_cond_expr(cns)
                            for cns in bounds_checks] + list(pred_checks)

                    prev_result = prev_gen_code(inner_codegen_state)

                    return [wrap_in_if(
                        inner_codegen_state,
                        condition_exprs,
                        merge_codegen_results(codegen_state, prev_result))]

                cannot_vectorize = False
                if new_codegen_state.vectorization_info is not None:
                    from loopy.isl_helpers import obj_involves_variable
                    for cond in bounds_checks:
                        if obj_involves_variable(
                                cond,
                                new_codegen_state.vectorization_info.iname):
                            cannot_vectorize = True
                            break

                if cannot_vectorize:
                    def gen_code_wrapper(inner_codegen_state):
                        # gen_code returns a list, but this needs to return a
                        # GeneratedCode instance.

                        return gen_code(inner_codegen_state)

                    result = [new_codegen_state.unvectorize(gen_code_wrapper)]
                else:
                    result = gen_code(new_codegen_state)

            else:
                result = gen_code(new_codegen_state)

        return result + build_insn_group(
                sched_index_info_entries[group_length:], codegen_state)

    # }}}

    insn_group = build_insn_group(sched_index_info_entries, codegen_state)
    return merge_codegen_results(
            codegen_state,
            insn_group)


# vim: foldmethod=marker

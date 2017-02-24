from __future__ import division, absolute_import

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


import six
from loopy.diagnostic import (
        LoopyError, WriteRaceConditionWarning, warn_with_kernel,
        LoopyAdvisory, DependencyTypeInferenceFailure)

import islpy as isl

from pytools.persistent_dict import PersistentDict

from loopy.tools import LoopyKeyBuilder
from loopy.version import DATA_MODEL_VERSION
from loopy.kernel.data import make_assignment
# for the benefit of loopy.statistics, for now
from loopy.type_inference import infer_unknown_types

import logging
logger = logging.getLogger(__name__)


# {{{ prepare for caching

def prepare_for_caching(kernel):
    import loopy as lp
    new_args = []

    for arg in kernel.args:
        dtype = arg.dtype
        if dtype is not None and dtype is not lp.auto:
            dtype = dtype.with_target(kernel.target)

        new_args.append(arg.copy(dtype=dtype))

    new_temporary_variables = {}
    for name, temp in six.iteritems(kernel.temporary_variables):
        dtype = temp.dtype
        if dtype is not None and dtype is not lp.auto:
            dtype = dtype.with_target(kernel.target)

        new_temporary_variables[name] = temp.copy(dtype=dtype)

    kernel = kernel.copy(
            args=new_args,
            temporary_variables=new_temporary_variables)

    return kernel

# }}}


# {{{ check for writes to predicates

def check_for_writes_to_predicates(kernel):
    from loopy.symbolic import get_dependencies
    for insn in kernel.instructions:
        pred_vars = (
                frozenset.union(
                    *(get_dependencies(pred) for pred in insn.predicates))
                if insn.predicates else frozenset())
        written_pred_vars = frozenset(insn.assignee_var_names()) & pred_vars
        if written_pred_vars:
            raise LoopyError("In instruction '%s': may not write to "
                    "variable(s) '%s' involved in the instruction's predicates"
                    % (insn.id, ", ".join(written_pred_vars)))

# }}}


# {{{ check reduction iname uniqueness

def check_reduction_iname_uniqueness(kernel):
    iname_to_reduction_count = {}
    iname_to_nonsimultaneous_reduction_count = {}

    def map_reduction(expr, rec):
        rec(expr.expr)
        for iname in expr.inames:
            iname_to_reduction_count[iname] = (
                    iname_to_reduction_count.get(iname, 0) + 1)
            if not expr.allow_simultaneous:
                iname_to_nonsimultaneous_reduction_count[iname] = (
                        iname_to_nonsimultaneous_reduction_count.get(iname, 0) + 1)

        return expr

    from loopy.symbolic import ReductionCallbackMapper
    cb_mapper = ReductionCallbackMapper(map_reduction)

    for insn in kernel.instructions:
        insn.with_transformed_expressions(cb_mapper)

    for iname, count in six.iteritems(iname_to_reduction_count):
        nonsimul_count = iname_to_nonsimultaneous_reduction_count.get(iname, 0)

        if nonsimul_count and count > 1:
            raise LoopyError("iname '%s' used in more than one reduction. "
                    "(%d of them, to be precise.) "
                    "Since this usage can easily cause loop scheduling "
                    "problems, this is prohibited by default. "
                    "Use loopy.make_reduction_inames_unique() to fix this. "
                    "If you are sure that this is OK, write the reduction "
                    "as 'simul_reduce(...)' instead of 'reduce(...)'"
                    % (iname, count))

# }}}


# {{{ decide temporary scope

def _get_compute_inames_tagged(kernel, insn, tag_base):
    return set(iname
            for iname in kernel.insn_inames(insn.id)
            if isinstance(kernel.iname_to_tag.get(iname), tag_base))


def _get_assignee_inames_tagged(kernel, insn, tag_base, tv_names):
    return set(iname
            for aname, adeps in zip(
                insn.assignee_var_names(),
                insn.assignee_subscript_deps())
            for iname in adeps & kernel.all_inames()
            if aname in tv_names
            if isinstance(kernel.iname_to_tag.get(iname), tag_base))


def find_temporary_scope(kernel):
    logger.debug("%s: find temporary scope" % kernel.name)

    new_temp_vars = {}
    from loopy.kernel.data import (LocalIndexTagBase, GroupIndexTag,
            temp_var_scope)
    import loopy as lp

    writers = kernel.writer_map()

    base_storage_to_aliases = {}

    kernel_var_names = kernel.all_variable_names(include_temp_storage=False)

    for temp_var in six.itervalues(kernel.temporary_variables):
        if temp_var.base_storage is not None:
            # no nesting allowed
            if temp_var.base_storage in kernel_var_names:
                raise LoopyError("base_storage for temporary '%s' is '%s', "
                        "which is an existing variable name"
                        % (temp_var.name, temp_var.base_storage))

            base_storage_to_aliases.setdefault(
                    temp_var.base_storage, []).append(temp_var.name)

    for temp_var in six.itervalues(kernel.temporary_variables):
        # Only fill out for variables that do not yet know if they're
        # local. (I.e. those generated by implicit temporary generation.)

        if temp_var.scope is not lp.auto:
            new_temp_vars[temp_var.name] = temp_var
            continue

        tv_names = (frozenset([temp_var.name])
                | frozenset(base_storage_to_aliases.get(temp_var.base_storage, [])))
        my_writers = writers.get(temp_var.name, frozenset())
        if temp_var.base_storage is not None:
            for alias in base_storage_to_aliases.get(temp_var.base_storage, []):
                my_writers = my_writers | writers.get(alias, frozenset())

        desired_scope_per_insn = []
        for insn_id in my_writers:
            insn = kernel.id_to_insn[insn_id]

            # A write race will emerge if:
            #
            # - the variable is local
            #   and
            # - the instruction is run across more inames (locally) parallel
            #   than are reflected in the assignee indices.

            locparallel_compute_inames = _get_compute_inames_tagged(
                    kernel, insn, LocalIndexTagBase)

            locparallel_assignee_inames = _get_assignee_inames_tagged(
                    kernel, insn, LocalIndexTagBase, tv_names)

            grpparallel_compute_inames = _get_compute_inames_tagged(
                    kernel, insn, GroupIndexTag)

            grpparallel_assignee_inames = _get_assignee_inames_tagged(
                    kernel, insn, GroupIndexTag, temp_var.name)

            assert locparallel_assignee_inames <= locparallel_compute_inames
            assert grpparallel_assignee_inames <= grpparallel_compute_inames

            desired_scope = temp_var_scope.PRIVATE
            for iname_descr, scope_descr, apin, cpin, scope in [
                    ("local", "local", locparallel_assignee_inames,
                        locparallel_compute_inames, temp_var_scope.LOCAL),
                    ("group", "global", grpparallel_assignee_inames,
                        grpparallel_compute_inames, temp_var_scope.GLOBAL),
                    ]:

                if (apin != cpin and bool(apin)):
                    warn_with_kernel(
                            kernel,
                            "write_race_%s(%s)" % (scope_descr, insn_id),
                            "instruction '%s' looks invalid: "
                            "it assigns to indices based on %s IDs, but "
                            "its temporary '%s' cannot be made %s because "
                            "a write race across the iname(s) '%s' would emerge. "
                            "(Do you need to add an extra iname to your prefetch?)"
                            % (insn_id, iname_descr, temp_var.name, scope_descr,
                                ", ".join(cpin - apin)),
                            WriteRaceConditionWarning)

                if (apin == cpin

                        # doesn't want to be in this scope if there aren't any
                        # parallel inames of that kind:
                        and bool(cpin)):
                    desired_scope = max(desired_scope, scope)

            desired_scope_per_insn.append(desired_scope)

        if not desired_scope_per_insn:
            if temp_var.initializer is None:
                warn_with_kernel(kernel, "temp_to_write(%s)" % temp_var.name,
                        "temporary variable '%s' never written, eliminating"
                        % temp_var.name, LoopyAdvisory)
            else:
                raise LoopyError("temporary variable '%s': never written, "
                        "cannot automatically determine scope"
                        % temp_var.name)

            continue

        overall_scope = max(desired_scope_per_insn)

        from pytools import all
        if not all(iscope == overall_scope for iscope in desired_scope_per_insn):
            raise LoopyError("not all instructions agree on the "
                    "the desired scope (private/local/global) of  the "
                    "temporary '%s'" % temp_var.name)

        new_temp_vars[temp_var.name] = temp_var.copy(scope=overall_scope)

    return kernel.copy(temporary_variables=new_temp_vars)

# }}}


# {{{ rewrite reduction to imperative form

def realize_reduction(kernel, insn_id_filter=None, unknown_types_ok=True):
    """Rewrites reductions into their imperative form. With *insn_id_filter*
    specified, operate only on the instruction with an instruction id matching
    *insn_id_filter*.

    If *insn_id_filter* is given, only the outermost level of reductions will be
    expanded, inner reductions will be left alone (because they end up in a new
    instruction with a different ID, which doesn't match the filter).

    If *insn_id_filter* is not given, all reductions in all instructions will
    be realized.
    """

    logger.debug("%s: realize reduction" % kernel.name)

    new_insns = []
    new_iname_tags = {}

    insn_id_gen = kernel.get_instruction_id_generator()

    var_name_gen = kernel.get_var_name_generator()
    new_temporary_variables = kernel.temporary_variables.copy()

    from loopy.type_inference import TypeInferenceMapper
    type_inf_mapper = TypeInferenceMapper(kernel)

    # {{{ sequential

    def map_reduction_seq(expr, rec, nresults, arg_dtype,
            reduction_dtypes):
        outer_insn_inames = temp_kernel.insn_inames(insn)

        from pymbolic import var
        acc_var_names = [
                var_name_gen("acc_"+"_".join(expr.inames))
                for i in range(nresults)]
        acc_vars = tuple(var(n) for n in acc_var_names)

        from loopy.kernel.data import TemporaryVariable, temp_var_scope

        for name, dtype in zip(acc_var_names, reduction_dtypes):
            new_temporary_variables[name] = TemporaryVariable(
                    name=name,
                    shape=(),
                    dtype=dtype,
                    scope=temp_var_scope.PRIVATE)

        init_id = insn_id_gen(
                "%s_%s_init" % (insn.id, "_".join(expr.inames)))

        init_insn = make_assignment(
                id=init_id,
                assignees=acc_vars,
                within_inames=outer_insn_inames - frozenset(expr.inames),
                within_inames_is_final=insn.within_inames_is_final,
                depends_on=frozenset(),
                expression=expr.operation.neutral_element(arg_dtype, expr.inames))

        generated_insns.append(init_insn)

        update_id = insn_id_gen(
                based_on="%s_%s_update" % (insn.id, "_".join(expr.inames)))

        update_insn_iname_deps = temp_kernel.insn_inames(insn) | set(expr.inames)
        if insn.within_inames_is_final:
            update_insn_iname_deps = insn.within_inames | set(expr.inames)

        reduction_insn = make_assignment(
                id=update_id,
                assignees=acc_vars,
                expression=expr.operation(
                    arg_dtype,
                    acc_vars if len(acc_vars) > 1 else acc_vars[0],
                    expr.expr, expr.inames),
                depends_on=frozenset([init_insn.id]) | insn.depends_on,
                within_inames=update_insn_iname_deps,
                within_inames_is_final=insn.within_inames_is_final)

        generated_insns.append(reduction_insn)

        new_insn_add_depends_on.add(reduction_insn.id)

        if nresults == 1:
            assert len(acc_vars) == 1
            return acc_vars[0]
        else:
            return acc_vars

    # }}}

    # {{{ local-parallel

    def _get_int_iname_size(iname):
        from loopy.isl_helpers import static_max_of_pw_aff
        from loopy.symbolic import pw_aff_to_expr
        size = pw_aff_to_expr(
                static_max_of_pw_aff(
                    kernel.get_iname_bounds(iname).size,
                    constants_only=True))
        assert isinstance(size, six.integer_types)
        return size

    def _make_slab_set(iname, size):
        v = isl.make_zero_and_vars([iname])
        bs, = (
                v[0].le_set(v[iname])
                &
                v[iname].lt_set(v[0] + size)).get_basic_sets()
        return bs

    def map_reduction_local(expr, rec, nresults, arg_dtype,
            reduction_dtypes):
        red_iname, = expr.inames

        size = _get_int_iname_size(red_iname)

        outer_insn_inames = temp_kernel.insn_inames(insn)

        from loopy.kernel.data import LocalIndexTagBase
        outer_local_inames = tuple(
                oiname
                for oiname in outer_insn_inames
                if isinstance(
                    kernel.iname_to_tag.get(oiname),
                    LocalIndexTagBase))

        from pymbolic import var
        outer_local_iname_vars = tuple(
                var(oiname) for oiname in outer_local_inames)

        outer_local_iname_sizes = tuple(
                _get_int_iname_size(oiname)
                for oiname in outer_local_inames)

        # {{{ add separate iname to carry out the reduction

        # Doing this sheds any odd conditionals that may be active
        # on our red_iname.

        base_exec_iname = var_name_gen("red_"+red_iname)
        domains.append(_make_slab_set(base_exec_iname, size))
        new_iname_tags[base_exec_iname] = kernel.iname_to_tag[red_iname]

        # }}}

        neutral_var_names = [
                var_name_gen("neutral_"+red_iname)
                for i in range(nresults)]
        acc_var_names = [
                var_name_gen("acc_"+red_iname)
                for i in range(nresults)]
        acc_vars = tuple(var(n) for n in acc_var_names)

        from loopy.kernel.data import TemporaryVariable, temp_var_scope
        for name, dtype in zip(acc_var_names, reduction_dtypes):
            new_temporary_variables[name] = TemporaryVariable(
                    name=name,
                    shape=outer_local_iname_sizes + (size,),
                    dtype=dtype,
                    scope=temp_var_scope.LOCAL)
        for name, dtype in zip(neutral_var_names, reduction_dtypes):
            new_temporary_variables[name] = TemporaryVariable(
                    name=name,
                    shape=(),
                    dtype=dtype,
                    scope=temp_var_scope.PRIVATE)

        base_iname_deps = outer_insn_inames - frozenset(expr.inames)

        neutral = expr.operation.neutral_element(arg_dtype, expr.inames)

        init_id = insn_id_gen("%s_%s_init" % (insn.id, red_iname))
        init_insn = make_assignment(
                id=init_id,
                assignees=tuple(
                    acc_var[outer_local_iname_vars + (var(base_exec_iname),)]
                    for acc_var in acc_vars),
                expression=neutral,
                within_inames=base_iname_deps | frozenset([base_exec_iname]),
                within_inames_is_final=insn.within_inames_is_final,
                depends_on=frozenset())
        generated_insns.append(init_insn)

        def _strip_if_scalar(c):
            if len(acc_vars) == 1:
                return c[0]
            else:
                return c

        init_neutral_id = insn_id_gen("%s_%s_init_neutral" % (insn.id, red_iname))
        init_neutral_insn = make_assignment(
                id=init_neutral_id,
                assignees=tuple(var(nvn) for nvn in neutral_var_names),
                expression=neutral,
                within_inames=base_iname_deps | frozenset([base_exec_iname]),
                within_inames_is_final=insn.within_inames_is_final,
                depends_on=frozenset())
        generated_insns.append(init_neutral_insn)

        transfer_id = insn_id_gen("%s_%s_transfer" % (insn.id, red_iname))
        transfer_insn = make_assignment(
                id=transfer_id,
                assignees=tuple(
                    acc_var[outer_local_iname_vars + (var(red_iname),)]
                    for acc_var in acc_vars),
                expression=expr.operation(
                    arg_dtype,
                    _strip_if_scalar(tuple(var(nvn) for nvn in neutral_var_names)),
                    expr.expr, expr.inames),
                within_inames=(
                    (outer_insn_inames - frozenset(expr.inames))
                    | frozenset([red_iname])),
                within_inames_is_final=insn.within_inames_is_final,
                depends_on=frozenset([init_id, init_neutral_id]) | insn.depends_on,
                no_sync_with=frozenset([(init_id, "any")]))
        generated_insns.append(transfer_insn)

        cur_size = 1
        while cur_size < size:
            cur_size *= 2

        prev_id = transfer_id
        bound = size

        istage = 0
        while cur_size > 1:

            new_size = cur_size // 2
            assert new_size * 2 == cur_size

            stage_exec_iname = var_name_gen("red_%s_s%d" % (red_iname, istage))
            domains.append(_make_slab_set(stage_exec_iname, bound-new_size))
            new_iname_tags[stage_exec_iname] = kernel.iname_to_tag[red_iname]

            stage_id = insn_id_gen("red_%s_stage_%d" % (red_iname, istage))
            stage_insn = make_assignment(
                    id=stage_id,
                    assignees=tuple(
                        acc_var[outer_local_iname_vars + (var(stage_exec_iname),)]
                        for acc_var in acc_vars),
                    expression=expr.operation(
                        arg_dtype,
                        _strip_if_scalar(tuple(
                            acc_var[
                                outer_local_iname_vars + (var(stage_exec_iname),)]
                            for acc_var in acc_vars)),
                        _strip_if_scalar(tuple(
                            acc_var[
                                outer_local_iname_vars + (
                                    var(stage_exec_iname) + new_size,)]
                            for acc_var in acc_vars)),
                        expr.inames),
                    within_inames=(
                        base_iname_deps | frozenset([stage_exec_iname])),
                    within_inames_is_final=insn.within_inames_is_final,
                    depends_on=frozenset([prev_id]),
                    )

            generated_insns.append(stage_insn)
            prev_id = stage_id

            cur_size = new_size
            bound = cur_size
            istage += 1

        new_insn_add_depends_on.add(prev_id)
        new_insn_add_no_sync_with.add((prev_id, "any"))
        new_insn_add_within_inames.add(stage_exec_iname or base_exec_iname)

        if nresults == 1:
            assert len(acc_vars) == 1
            return acc_vars[0][outer_local_iname_vars + (0,)]
        else:
            return [acc_var[outer_local_iname_vars + (0,)] for acc_var in acc_vars]
    # }}}

    # {{{ seq/par dispatch

    def map_reduction(expr, rec, nresults=1):
        # Only expand one level of reduction at a time, going from outermost to
        # innermost. Otherwise we get the (iname + insn) dependencies wrong.

        try:
            arg_dtype = type_inf_mapper(expr.expr)
        except DependencyTypeInferenceFailure:
            if unknown_types_ok:
                arg_dtype = lp.auto

                reduction_dtypes = (lp.auto,)*nresults

            else:
                raise LoopyError("failed to determine type of accumulator for "
                        "reduction '%s'" % expr)
        else:
            arg_dtype = arg_dtype.with_target(kernel.target)

            reduction_dtypes = expr.operation.result_dtypes(
                        kernel, arg_dtype, expr.inames)
            reduction_dtypes = tuple(
                    dt.with_target(kernel.target) for dt in reduction_dtypes)

        outer_insn_inames = temp_kernel.insn_inames(insn)
        bad_inames = frozenset(expr.inames) & outer_insn_inames
        if bad_inames:
            raise LoopyError("reduction used within loop(s) that it was "
                    "supposed to reduce over: " + ", ".join(bad_inames))

        n_sequential = 0
        n_local_par = 0

        from loopy.kernel.data import (
                LocalIndexTagBase, UnrolledIlpTag, UnrollTag, VectorizeTag,
                ParallelTag)
        for iname in expr.inames:
            iname_tag = kernel.iname_to_tag.get(iname)

            if isinstance(iname_tag, (UnrollTag, UnrolledIlpTag)):
                # These are nominally parallel, but we can live with
                # them as sequential.
                n_sequential += 1

            elif isinstance(iname_tag, LocalIndexTagBase):
                n_local_par += 1

            elif isinstance(iname_tag, (ParallelTag, VectorizeTag)):
                raise LoopyError("the only form of parallelism supported "
                        "by reductions is 'local'--found iname '%s' "
                        "tagged '%s'"
                        % (iname, type(iname_tag).__name__))

            else:
                n_sequential += 1

        if n_local_par and n_sequential:
            raise LoopyError("Reduction over '%s' contains both parallel and "
                    "sequential inames. It must be split "
                    "(using split_reduction_{in,out}ward) "
                    "before code generation."
                    % ", ".join(expr.inames))

        if n_local_par > 1:
            raise LoopyError("Reduction over '%s' contains more than"
                    "one parallel iname. It must be split "
                    "(using split_reduction_{in,out}ward) "
                    "before code generation."
                    % ", ".join(expr.inames))

        if n_sequential:
            assert n_local_par == 0
            return map_reduction_seq(expr, rec, nresults, arg_dtype,
                    reduction_dtypes)
        elif n_local_par:
            return map_reduction_local(expr, rec, nresults, arg_dtype,
                    reduction_dtypes)
        else:
            from loopy.diagnostic import warn_with_kernel
            warn_with_kernel(kernel, "empty_reduction",
                    "Empty reduction found (no inames to reduce over). "
                    "Eliminating.")

            return expr.expr

    # }}}

    from loopy.symbolic import ReductionCallbackMapper
    cb_mapper = ReductionCallbackMapper(map_reduction)

    insn_queue = kernel.instructions[:]
    insn_id_replacements = {}
    domains = kernel.domains[:]

    temp_kernel = kernel

    import loopy as lp
    while insn_queue:
        new_insn_add_depends_on = set()
        new_insn_add_no_sync_with = set()
        new_insn_add_within_inames = set()

        generated_insns = []

        insn = insn_queue.pop(0)

        if insn_id_filter is not None and insn.id != insn_id_filter \
                or not isinstance(insn, lp.MultiAssignmentBase):
            new_insns.append(insn)
            continue

        nresults = len(insn.assignees)

        # Run reduction expansion.
        from loopy.symbolic import Reduction
        if isinstance(insn.expression, Reduction) and nresults > 1:
            new_expressions = cb_mapper(insn.expression, nresults=nresults)
        else:
            new_expressions = (cb_mapper(insn.expression),)

        if generated_insns:
            # An expansion happened, so insert the generated stuff plus
            # ourselves back into the queue.

            kwargs = insn.get_copy_kwargs(
                    depends_on=insn.depends_on
                    | frozenset(new_insn_add_depends_on),
                    no_sync_with=insn.no_sync_with
                    | frozenset(new_insn_add_no_sync_with),
                    within_inames=(
                        temp_kernel.insn_inames(insn)
                        | new_insn_add_within_inames))

            kwargs.pop("id")
            kwargs.pop("expression")
            kwargs.pop("assignee", None)
            kwargs.pop("assignees", None)
            kwargs.pop("temp_var_type", None)
            kwargs.pop("temp_var_types", None)

            if isinstance(insn.expression, Reduction) and nresults > 1:
                replacement_insns = [
                        lp.Assignment(
                            id=insn_id_gen(insn.id),
                            assignee=assignee,
                            expression=new_expr,
                            **kwargs)
                        for assignee, new_expr in zip(
                            insn.assignees, new_expressions)]

            else:
                new_expr, = new_expressions
                replacement_insns = [
                        make_assignment(
                            id=insn_id_gen(insn.id),
                            assignees=insn.assignees,
                            expression=new_expr,
                            **kwargs)
                        ]

            insn_id_replacements[insn.id] = [
                    rinsn.id for rinsn in replacement_insns]

            insn_queue = generated_insns + replacement_insns + insn_queue

            # The reduction expander needs an up-to-date kernel
            # object to find dependencies. Keep temp_kernel up-to-date.

            temp_kernel = kernel.copy(
                    instructions=new_insns + insn_queue,
                    temporary_variables=new_temporary_variables,
                    domains=domains)
            temp_kernel = lp.replace_instruction_ids(
                    temp_kernel, insn_id_replacements)

        else:
            # nothing happened, we're done with insn
            assert not new_insn_add_depends_on

            new_insns.append(insn)

    kernel = kernel.copy(
            instructions=new_insns,
            temporary_variables=new_temporary_variables,
            domains=domains)

    kernel = lp.replace_instruction_ids(kernel, insn_id_replacements)

    kernel = lp.tag_inames(kernel, new_iname_tags)

    return kernel

# }}}


# {{{ find idempotence ("boostability") of instructions

def find_idempotence(kernel):
    logger.debug("%s: idempotence" % kernel.name)

    writer_map = kernel.writer_map()

    arg_names = set(arg.name for arg in kernel.args)

    var_names = arg_names | set(six.iterkeys(kernel.temporary_variables))

    reads_map = dict(
            (insn.id, insn.read_dependency_names() & var_names)
            for insn in kernel.instructions)

    from collections import defaultdict
    dep_graph = defaultdict(lambda: set())

    for insn in kernel.instructions:
        dep_graph[insn.id] = set(writer_id
                for var in reads_map[insn.id]
                for writer_id in writer_map.get(var, set()))

    # Find SCCs of dep_graph. These are used for checking if the instruction is
    # in a dependency cycle.
    from loopy.tools import compute_sccs

    sccs = dict((item, scc)
            for scc in compute_sccs(dep_graph)
            for item in scc)

    non_idempotently_updated_vars = set()

    new_insns = []
    for insn in kernel.instructions:
        boostable = len(sccs[insn.id]) == 1 and insn.id not in dep_graph[insn.id]

        if not boostable:
            non_idempotently_updated_vars.update(
                    insn.assignee_var_names())

        new_insns.append(insn.copy(boostable=boostable))

    # {{{ remove boostability from isns that access non-idempotently updated vars

    new2_insns = []
    for insn in new_insns:
        if insn.boostable and bool(
                non_idempotently_updated_vars & insn.dependency_names()):
            new2_insns.append(insn.copy(boostable=False))
        else:
            new2_insns.append(insn)

    # }}}

    return kernel.copy(instructions=new2_insns)

# }}}


# {{{ limit boostability

def limit_boostability(kernel):
    """Finds out which other inames an instruction's inames occur with
    and then limits boostability to just those inames.
    """

    logger.debug("%s: limit boostability" % kernel.name)

    iname_occurs_with = {}
    for insn in kernel.instructions:
        insn_inames = kernel.insn_inames(insn)
        for iname in insn_inames:
            iname_occurs_with.setdefault(iname, set()).update(insn_inames)

    iname_use_counts = {}
    for insn in kernel.instructions:
        for iname in kernel.insn_inames(insn):
            iname_use_counts[iname] = iname_use_counts.get(iname, 0) + 1

    single_use_inames = set(iname for iname, uc in six.iteritems(iname_use_counts)
            if uc == 1)

    new_insns = []
    for insn in kernel.instructions:
        if insn.boostable is None:
            raise LoopyError("insn '%s' has undetermined boostability" % insn.id)
        elif insn.boostable:
            boostable_into = set()
            for iname in kernel.insn_inames(insn):
                boostable_into.update(iname_occurs_with[iname])

            boostable_into -= kernel.insn_inames(insn) | single_use_inames

            # Even if boostable_into is empty, leave boostable flag on--it is used
            # for boosting into unused hw axes.

            insn = insn.copy(boostable_into=boostable_into)
        else:
            insn = insn.copy(boostable_into=set())

        new_insns.append(insn)

    return kernel.copy(instructions=new_insns)

# }}}


preprocess_cache = PersistentDict("loopy-preprocess-cache-v2-"+DATA_MODEL_VERSION,
        key_builder=LoopyKeyBuilder())


def preprocess_kernel(kernel, device=None):
    if device is not None:
        from warnings import warn
        warn("passing 'device' to preprocess_kernel() is deprecated",
                DeprecationWarning, stacklevel=2)

    from loopy.kernel import kernel_state
    if kernel.state >= kernel_state.PREPROCESSED:
        return kernel

    # {{{ cache retrieval

    from loopy import CACHING_ENABLED
    if CACHING_ENABLED:
        input_kernel = kernel

        try:
            result = preprocess_cache[kernel]
            logger.debug("%s: preprocess cache hit" % kernel.name)
            return result
        except KeyError:
            pass

    # }}}

    logger.info("%s: preprocess start" % kernel.name)

    from loopy.check import check_identifiers_in_subst_rules
    check_identifiers_in_subst_rules(kernel)

    # {{{ check that there are no l.auto-tagged inames

    from loopy.kernel.data import AutoLocalIndexTagBase
    for iname, tag in six.iteritems(kernel.iname_to_tag):
        if (isinstance(tag, AutoLocalIndexTagBase)
                 and iname in kernel.all_inames()):
            raise LoopyError("kernel with automatically-assigned "
                    "local axes passed to preprocessing")

    # }}}

    from loopy.transform.subst import expand_subst
    kernel = expand_subst(kernel)

    # Ordering restriction:
    # Type inference and reduction iname uniqueness don't handle substitutions.
    # Get them out of the way.

    kernel = infer_unknown_types(kernel, expect_completion=False)

    check_for_writes_to_predicates(kernel)
    check_reduction_iname_uniqueness(kernel)

    from loopy.kernel.creation import apply_single_writer_depencency_heuristic
    kernel = apply_single_writer_depencency_heuristic(kernel)

    # Ordering restrictions:
    #
    # - realize_reduction must happen after type inference because it needs
    #   to be able to determine the types of the reduced expressions.
    #
    # - realize_reduction must happen after default dependencies are added
    #   because it manipulates the depends_on field, which could prevent
    #   defaults from being applied.

    kernel = realize_reduction(kernel, unknown_types_ok=False)

    # Ordering restriction:
    # add_axes_to_temporaries_for_ilp because reduction accumulators
    # need to be duplicated by this.

    from loopy.transform.ilp import add_axes_to_temporaries_for_ilp_and_vec
    kernel = add_axes_to_temporaries_for_ilp_and_vec(kernel)

    kernel = find_temporary_scope(kernel)

    # boostability should be removed in 2017.x.
    kernel = find_idempotence(kernel)
    kernel = limit_boostability(kernel)

    kernel = kernel.target.preprocess(kernel)

    logger.info("%s: preprocess done" % kernel.name)

    kernel = kernel.copy(
            state=kernel_state.PREPROCESSED)

    # {{{ prepare for caching

    # PicklableDtype instances for example need to know the target they're working
    # towards in order to pickle and unpickle them. This is the first pass that
    # uses caching, so we need to be ready to pickle. This means propagating
    # this target information.

    if CACHING_ENABLED:
        input_kernel = prepare_for_caching(input_kernel)

    kernel = prepare_for_caching(kernel)

    # }}}

    if CACHING_ENABLED:
        preprocess_cache[input_kernel] = kernel

    return kernel

# vim: foldmethod=marker

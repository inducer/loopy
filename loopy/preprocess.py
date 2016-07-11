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


# {{{ infer types

def _infer_var_type(kernel, var_name, type_inf_mapper, subst_expander):
    if var_name in kernel.all_params():
        return kernel.index_dtype, []

    def debug(s):
        logger.debug("%s: %s" % (kernel.name, s))

    dtypes = []

    import loopy as lp

    symbols_with_unavailable_types = []

    from loopy.diagnostic import DependencyTypeInferenceFailure
    for writer_insn_id in kernel.writer_map().get(var_name, []):
        writer_insn = kernel.id_to_insn[writer_insn_id]
        if not isinstance(writer_insn, lp.MultiAssignmentBase):
            continue

        expr = subst_expander(writer_insn.expression)

        try:
            debug("             via expr %s" % expr)
            if isinstance(writer_insn, lp.Assignment):
                result = type_inf_mapper(expr)
            elif isinstance(writer_insn, lp.CallInstruction):
                result_dtypes = type_inf_mapper(expr, multiple_types_ok=True)

                result = None
                for assignee, comp_dtype in zip(
                        writer_insn.assignee_var_names(), result_dtypes):
                    if assignee == var_name:
                        result = comp_dtype
                        break

                assert result is not None

            debug("             result: %s" % result)

            dtypes.append(result)

        except DependencyTypeInferenceFailure as e:
            debug("             failed: %s" % e)
            symbols_with_unavailable_types.append(e.symbol)

    if not dtypes:
        return None, symbols_with_unavailable_types

    result = type_inf_mapper.combine(dtypes)

    return result, []


class _DictUnionView:
    def __init__(self, children):
        self.children = children

    def get(self, key):
        try:
            return self[key]
        except KeyError:
            return None

    def __getitem__(self, key):
        for ch in self.children:
            try:
                return ch[key]
            except KeyError:
                pass

        raise KeyError(key)


def infer_unknown_types(kernel, expect_completion=False):
    """Infer types on temporaries and arguments."""

    logger.debug("%s: infer types" % kernel.name)

    def debug(s):
        logger.debug("%s: %s" % (kernel.name, s))

    unexpanded_kernel = kernel
    if kernel.substitutions:
        from loopy.transform.subst import expand_subst
        kernel = expand_subst(kernel)

    new_temp_vars = kernel.temporary_variables.copy()
    new_arg_dict = kernel.arg_dict.copy()

    # {{{ fill queue

    # queue contains temporary variables
    queue = []

    import loopy as lp
    for tv in six.itervalues(kernel.temporary_variables):
        if tv.dtype is lp.auto:
            queue.append(tv)

    for arg in kernel.args:
        if arg.dtype is None:
            queue.append(arg)

    # }}}

    from loopy.expression import TypeInferenceMapper
    type_inf_mapper = TypeInferenceMapper(kernel,
            _DictUnionView([
                new_temp_vars,
                new_arg_dict
                ]))

    from loopy.symbolic import SubstitutionRuleExpander
    subst_expander = SubstitutionRuleExpander(kernel.substitutions)

    # {{{ work on type inference queue

    from loopy.kernel.data import TemporaryVariable, KernelArgument

    failed_names = set()
    while queue:
        item = queue.pop(0)

        debug("inferring type for %s %s" % (type(item).__name__, item.name))

        result, symbols_with_unavailable_types = \
                _infer_var_type(kernel, item.name, type_inf_mapper, subst_expander)

        failed = result is None
        if not failed:
            debug("     success: %s" % result)
            if isinstance(item, TemporaryVariable):
                new_temp_vars[item.name] = item.copy(dtype=result)
            elif isinstance(item, KernelArgument):
                new_arg_dict[item.name] = item.copy(dtype=result)
            else:
                raise LoopyError("unexpected item type in type inference")
        else:
            debug("     failure")

        if failed:
            if item.name in failed_names:
                # this item has failed before, give up.
                advice = ""
                if symbols_with_unavailable_types:
                    advice += (
                            " (need type of '%s'--check for missing arguments)"
                            % ", ".join(symbols_with_unavailable_types))

                if expect_completion:
                    raise LoopyError(
                            "could not determine type of '%s'%s"
                            % (item.name, advice))

                else:
                    # We're done here.
                    break

            # remember that this item failed
            failed_names.add(item.name)

            queue_names = set(qi.name for qi in queue)

            if queue_names == failed_names:
                # We did what we could...
                print(queue_names, failed_names, item.name)
                assert not expect_completion
                break

            # can't infer type yet, put back into queue
            queue.append(item)
        else:
            # we've made progress, reset failure markers
            failed_names = set()

    # }}}

    return unexpanded_kernel.copy(
            temporary_variables=new_temp_vars,
            args=[new_arg_dict[arg.name] for arg in kernel.args],
            )

# }}}


# {{{ decide temporary scope

def _get_compute_inames_tagged(kernel, insn, tag_base):
    return set(iname
            for iname in kernel.insn_inames(insn.id)
            if isinstance(kernel.iname_to_tag.get(iname), tag_base))


def _get_assignee_inames_tagged(kernel, insn, tag_base, tv_name):
    return set(iname
            for aname, adeps in zip(
                insn.assignee_var_names(),
                insn.assignee_subscript_deps())
            for iname in adeps & kernel.all_inames()
            if aname == tv_name
            if isinstance(kernel.iname_to_tag.get(iname), tag_base))


def find_temporary_scope(kernel):
    logger.debug("%s: mark local temporaries" % kernel.name)

    new_temp_vars = {}
    from loopy.kernel.data import (LocalIndexTagBase, GroupIndexTag,
            temp_var_scope)
    import loopy as lp

    writers = kernel.writer_map()

    for temp_var in six.itervalues(kernel.temporary_variables):
        # Only fill out for variables that do not yet know if they're
        # local. (I.e. those generated by implicit temporary generation.)

        if temp_var.scope is not lp.auto:
            new_temp_vars[temp_var.name] = temp_var
            continue

        my_writers = writers.get(temp_var.name, [])

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
                    kernel, insn, LocalIndexTagBase, temp_var.name)

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

                if (apin != cpin and bool(locparallel_assignee_inames)):
                    warn_with_kernel(kernel, "write_race_local(%s)" % insn_id,
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
                    break

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


# {{{ default dependencies

def add_default_dependencies(kernel):
    logger.debug("%s: default deps" % kernel.name)

    from loopy.transform.subst import expand_subst
    expanded_kernel = expand_subst(kernel)

    writer_map = kernel.writer_map()

    arg_names = set(arg.name for arg in kernel.args)

    var_names = arg_names | set(six.iterkeys(kernel.temporary_variables))

    dep_map = dict(
            (insn.id, insn.read_dependency_names() & var_names)
            for insn in expanded_kernel.instructions)

    new_insns = []
    for insn in kernel.instructions:
        if not insn.depends_on_is_final:
            auto_deps = set()

            # {{{ add automatic dependencies

            all_my_var_writers = set()
            for var in dep_map[insn.id]:
                var_writers = writer_map.get(var, set())
                all_my_var_writers |= var_writers

                if not var_writers and var not in arg_names:
                    tv = kernel.temporary_variables[var]
                    if tv.initializer is None:
                        warn_with_kernel(kernel, "read_no_write(%s)" % var,
                                "temporary variable '%s' is read, but never written."
                                % var)

                if len(var_writers) == 1:
                    auto_deps.update(
                            var_writers
                            - set([insn.id]))

            # }}}

            depends_on = insn.depends_on
            if depends_on is None:
                depends_on = frozenset()

            insn = insn.copy(depends_on=frozenset(auto_deps) | depends_on)

        new_insns.append(insn)

    return kernel.copy(instructions=new_insns)

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

    from loopy.expression import TypeInferenceMapper
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
                forced_iname_deps=outer_insn_inames - frozenset(expr.inames),
                forced_iname_deps_is_final=insn.forced_iname_deps_is_final,
                depends_on=frozenset(),
                expression=expr.operation.neutral_element(arg_dtype, expr.inames))

        generated_insns.append(init_insn)

        update_id = insn_id_gen(
                based_on="%s_%s_update" % (insn.id, "_".join(expr.inames)))

        update_insn_iname_deps = temp_kernel.insn_inames(insn) | set(expr.inames)
        if insn.forced_iname_deps_is_final:
            update_insn_iname_deps = insn.forced_iname_deps | set(expr.inames)

        reduction_insn = make_assignment(
                id=update_id,
                assignees=acc_vars,
                expression=expr.operation(
                    arg_dtype,
                    acc_vars if len(acc_vars) > 1 else acc_vars[0],
                    expr.expr, expr.inames),
                depends_on=frozenset([init_insn.id]) | insn.depends_on,
                forced_iname_deps=update_insn_iname_deps,
                forced_iname_deps_is_final=insn.forced_iname_deps_is_final)

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

        base_iname_deps = outer_insn_inames - frozenset(expr.inames)

        neutral = expr.operation.neutral_element(arg_dtype, expr.inames)

        init_id = insn_id_gen("%s_%s_init" % (insn.id, red_iname))
        init_insn = make_assignment(
                id=init_id,
                assignees=tuple(
                    acc_var[outer_local_iname_vars + (var(base_exec_iname),)]
                    for acc_var in acc_vars),
                expression=neutral,
                forced_iname_deps=base_iname_deps | frozenset([base_exec_iname]),
                forced_iname_deps_is_final=insn.forced_iname_deps_is_final,
                depends_on=frozenset())
        generated_insns.append(init_insn)

        transfer_id = insn_id_gen("%s_%s_transfer" % (insn.id, red_iname))
        transfer_insn = make_assignment(
                id=transfer_id,
                assignees=tuple(
                    acc_var[outer_local_iname_vars + (var(red_iname),)]
                    for acc_var in acc_vars),
                expression=expr.operation(
                    arg_dtype, neutral, expr.expr, expr.inames),
                forced_iname_deps=(
                    (outer_insn_inames - frozenset(expr.inames))
                    | frozenset([red_iname])),
                forced_iname_deps_is_final=insn.forced_iname_deps_is_final,
                depends_on=frozenset([init_id]) | insn.depends_on,
                no_sync_with=frozenset([init_id]))
        generated_insns.append(transfer_insn)

        def _strip_if_scalar(c):
            if len(acc_vars) == 1:
                return c[0]
            else:
                return c

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
                        _strip_if_scalar([
                            acc_var[
                                outer_local_iname_vars + (var(stage_exec_iname),)]
                            for acc_var in acc_vars]),
                        _strip_if_scalar([
                            acc_var[
                                outer_local_iname_vars + (
                                    var(stage_exec_iname) + new_size,)]
                            for acc_var in acc_vars]),
                        expr.inames),
                    forced_iname_deps=(
                        base_iname_deps | frozenset([stage_exec_iname])),
                    forced_iname_deps_is_final=insn.forced_iname_deps_is_final,
                    depends_on=frozenset([prev_id]),
                    )

            generated_insns.append(stage_insn)
            prev_id = stage_id

            cur_size = new_size
            bound = cur_size
            istage += 1

        new_insn_add_depends_on.add(prev_id)
        new_insn_add_no_sync_with.add(prev_id)
        new_insn_add_forced_iname_deps.add(stage_exec_iname or base_exec_iname)

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
        new_insn_add_forced_iname_deps = set()

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
                    forced_iname_deps=(
                        temp_kernel.insn_inames(insn)
                        | new_insn_add_forced_iname_deps))

            kwargs.pop("id")
            kwargs.pop("expression")
            kwargs.pop("assignee", None)
            kwargs.pop("assignees", None)
            kwargs.pop("temp_var_type", None)
            kwargs.pop("temp_var_types", None)

            replacement_insns = [
                    lp.Assignment(
                        id=insn_id_gen(insn.id),
                        assignee=assignee,
                        expression=new_expr,
                        **kwargs)
                    for assignee, new_expr in zip(insn.assignees, new_expressions)]

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

    non_idempotently_updated_vars = set()

    # FIXME: This can be made more efficient by simply starting
    # from all written variables and not even considering
    # instructions as the start of the first pass.

    new_insns = []
    for insn in kernel.instructions:
        all_my_var_writers = set()
        for var in reads_map[insn.id]:
            var_writers = writer_map.get(var, set())
            all_my_var_writers |= var_writers

        # {{{ find dependency loops, flag boostability

        while True:
            last_all_my_var_writers = all_my_var_writers

            for writer_insn_id in last_all_my_var_writers:
                for var in reads_map[writer_insn_id]:
                    all_my_var_writers = \
                            all_my_var_writers | writer_map.get(var, set())

            if last_all_my_var_writers == all_my_var_writers:
                break

        # }}}

        boostable = insn.id not in all_my_var_writers

        if not boostable:
            non_idempotently_updated_vars.update(
                    insn.assignee_var_names())

        insn = insn.copy(boostable=boostable)

        new_insns.append(insn)

    # {{{ remove boostability from isns that access non-idempotently updated vars

    new2_insns = []
    for insn in new_insns:
        accessed_vars = insn.dependency_names()
        boostable = insn.boostable and not bool(
                non_idempotently_updated_vars & accessed_vars)
        new2_insns.append(insn.copy(boostable=boostable))

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
    if kernel.state != kernel_state.INITIAL:
        raise LoopyError("cannot re-preprocess an already preprocessed "
                "kernel")

    # {{{ cache retrieval

    from loopy import CACHING_ENABLED
    if CACHING_ENABLED:
        input_kernel = kernel

        try:
            result = preprocess_cache[kernel]
            logger.info("%s: preprocess cache hit" % kernel.name)
            return result
        except KeyError:
            pass

    # }}}

    logger.info("%s: preprocess start" % kernel.name)

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

    check_reduction_iname_uniqueness(kernel)

    kernel = add_default_dependencies(kernel)

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

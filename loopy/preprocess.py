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
        LoopyError, WriteRaceConditionWarning, warn,
        LoopyAdvisory, DependencyTypeInferenceFailure)

from pytools.persistent_dict import PersistentDict
from loopy.tools import LoopyKeyBuilder
from loopy.version import DATA_MODEL_VERSION

import logging
logger = logging.getLogger(__name__)


# {{{ prepare for caching

def prepare_for_caching(kernel):
    import loopy as lp
    new_args = []

    for arg in kernel.args:
        dtype = arg.picklable_dtype
        if dtype is not None and dtype is not lp.auto:
            dtype = dtype.with_target(kernel.target)

        new_args.append(arg.copy(dtype=dtype))

    new_temporary_variables = {}
    for name, temp in six.iteritems(kernel.temporary_variables):
        dtype = temp.picklable_dtype
        if dtype is not None and dtype is not lp.auto:
            dtype = dtype.with_target(kernel.target)

        new_temporary_variables[name] = temp.copy(dtype=dtype)

    kernel = kernel.copy(
            args=new_args,
            temporary_variables=new_temporary_variables)

    return kernel

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
        if not isinstance(writer_insn, lp.Assignment):
            continue

        expr = subst_expander(writer_insn.expression)

        try:
            debug("             via expr %s" % expr)
            result = type_inf_mapper(expr)

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


# {{{ decide which temporaries are local

def mark_local_temporaries(kernel):
    logger.debug("%s: mark local temporaries" % kernel.name)

    new_temp_vars = {}
    from loopy.kernel.data import LocalIndexTagBase
    import loopy as lp

    writers = kernel.writer_map()

    from loopy.symbolic import get_dependencies

    for temp_var in six.itervalues(kernel.temporary_variables):
        # Only fill out for variables that do not yet know if they're
        # local. (I.e. those generated by implicit temporary generation.)

        if temp_var.is_local is not lp.auto:
            new_temp_vars[temp_var.name] = temp_var
            continue

        my_writers = writers.get(temp_var.name, [])

        wants_to_be_local_per_insn = []
        for insn_id in my_writers:
            insn = kernel.id_to_insn[insn_id]

            # A write race will emerge if:
            #
            # - the variable is local
            #   and
            # - the instruction is run across more inames (locally) parallel
            #   than are reflected in the assignee indices.

            locparallel_compute_inames = set(iname
                    for iname in kernel.insn_inames(insn_id)
                    if isinstance(kernel.iname_to_tag.get(iname), LocalIndexTagBase))

            locparallel_assignee_inames = set(iname
                    for _, assignee_indices in insn.assignees_and_indices()
                    for iname in get_dependencies(assignee_indices)
                        & kernel.all_inames()
                    if isinstance(kernel.iname_to_tag.get(iname), LocalIndexTagBase))

            assert locparallel_assignee_inames <= locparallel_compute_inames

            if (locparallel_assignee_inames != locparallel_compute_inames
                    and bool(locparallel_assignee_inames)):
                warn(kernel, "write_race_local(%s)" % insn_id,
                        "instruction '%s' looks invalid: "
                        "it assigns to indices based on local IDs, but "
                        "its temporary '%s' cannot be made local because "
                        "a write race across the iname(s) '%s' would emerge. "
                        "(Do you need to add an extra iname to your prefetch?)"
                        % (insn_id, temp_var.name, ", ".join(
                            locparallel_compute_inames
                            - locparallel_assignee_inames)),
                        WriteRaceConditionWarning)

            wants_to_be_local_per_insn.append(
                    locparallel_assignee_inames == locparallel_compute_inames

                    # doesn't want to be local if there aren't any
                    # parallel inames:
                    and bool(locparallel_compute_inames))

        if not wants_to_be_local_per_insn:
            warn(kernel, "temp_to_write(%s)" % temp_var.name,
                    "temporary variable '%s' never written, eliminating"
                    % temp_var.name, LoopyAdvisory)

            continue

        is_local = any(wants_to_be_local_per_insn)

        from pytools import all
        if not all(wtbl == is_local for wtbl in wants_to_be_local_per_insn):
            raise LoopyError("not all instructions agree on whether "
                    "temporary '%s' should be in local memory" % temp_var.name)

        new_temp_vars[temp_var.name] = temp_var.copy(is_local=is_local)

    return kernel.copy(temporary_variables=new_temp_vars)

# }}}


# {{{ default dependencies

def add_default_dependencies(kernel):
    logger.debug("%s: default deps" % kernel.name)

    writer_map = kernel.writer_map()

    arg_names = set(arg.name for arg in kernel.args)

    var_names = arg_names | set(six.iterkeys(kernel.temporary_variables))

    dep_map = dict(
            (insn.id, insn.read_dependency_names() & var_names)
            for insn in kernel.instructions)

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
                    warn(kernel, "read_no_write(%s)" % var,
                            "temporary variable '%s' is read, but never written."
                            % var)

                if len(var_writers) == 1:
                    auto_deps.update(var_writers - set([insn.id]))

            # }}}

            depends_on = insn.depends_on
            if depends_on is None:
                depends_on = frozenset()

            insn = insn.copy(depends_on=frozenset(auto_deps) | depends_on)

        new_insns.append(insn)

    return kernel.copy(instructions=new_insns)

# }}}


# {{{ rewrite reduction to imperative form

def realize_reduction(kernel, insn_id_filter=None):
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

    var_name_gen = kernel.get_var_name_generator()
    new_temporary_variables = kernel.temporary_variables.copy()

    from loopy.expression import TypeInferenceMapper
    type_inf_mapper = TypeInferenceMapper(kernel)

    def map_reduction(expr, rec):
        # Only expand one level of reduction at a time, going from outermost to
        # innermost. Otherwise we get the (iname + insn) dependencies wrong.

        from pymbolic import var

        target_var_name = var_name_gen("acc_"+"_".join(expr.inames))
        target_var = var(target_var_name)

        try:
            arg_dtype = type_inf_mapper(expr.expr)
        except DependencyTypeInferenceFailure:
            raise LoopyError("failed to determine type of accumulator for "
                    "reduction '%s'" % expr)

        from loopy.kernel.data import Assignment, TemporaryVariable

        new_temporary_variables[target_var_name] = TemporaryVariable(
                name=target_var_name,
                shape=(),
                dtype=expr.operation.result_dtype(
                    kernel.target, arg_dtype, expr.inames),
                is_local=False)

        outer_insn_inames = temp_kernel.insn_inames(insn)
        bad_inames = frozenset(expr.inames) & outer_insn_inames
        if bad_inames:
            raise LoopyError("reduction used within loop(s) that it was "
                    "supposed to reduce over: " + ", ".join(bad_inames))

        init_id = temp_kernel.make_unique_instruction_id(
                based_on="%s_%s_init" % (insn.id, "_".join(expr.inames)),
                extra_used_ids=set(i.id for i in generated_insns))

        init_insn = Assignment(
                id=init_id,
                assignee=target_var,
                forced_iname_deps=outer_insn_inames - frozenset(expr.inames),
                depends_on=frozenset(),
                expression=expr.operation.neutral_element(arg_dtype, expr.inames))

        generated_insns.append(init_insn)

        update_id = temp_kernel.make_unique_instruction_id(
                based_on="%s_%s_update" % (insn.id, "_".join(expr.inames)),
                extra_used_ids=set(i.id for i in generated_insns))

        reduction_insn = Assignment(
                id=update_id,
                assignee=target_var,
                expression=expr.operation(
                    arg_dtype, target_var, expr.expr, expr.inames),
                depends_on=frozenset([init_insn.id]) | insn.depends_on,
                forced_iname_deps=temp_kernel.insn_inames(insn) | set(expr.inames))

        generated_insns.append(reduction_insn)

        new_insn_depends_on.add(reduction_insn.id)

        return target_var

    from loopy.symbolic import ReductionCallbackMapper
    cb_mapper = ReductionCallbackMapper(map_reduction)

    insn_queue = kernel.instructions[:]

    temp_kernel = kernel

    import loopy as lp
    while insn_queue:
        new_insn_depends_on = set()
        generated_insns = []

        insn = insn_queue.pop(0)

        if insn_id_filter is not None and insn.id != insn_id_filter \
                or not isinstance(insn, lp.Assignment):
            new_insns.append(insn)
            continue

        # Run reduction expansion.
        new_expression = cb_mapper(insn.expression)

        if generated_insns:
            # An expansion happened, so insert the generated stuff plus
            # ourselves back into the queue.

            insn = insn.copy(
                        expression=new_expression,
                        depends_on=insn.depends_on
                        | frozenset(new_insn_depends_on),
                        forced_iname_deps=temp_kernel.insn_inames(insn))

            insn_queue = generated_insns + [insn] + insn_queue

            # The reduction expander needs an up-to-date kernel
            # object to find dependencies. Keep temp_kernel up-to-date.

            temp_kernel = kernel.copy(
                    instructions=new_insns + insn_queue,
                    temporary_variables=new_temporary_variables)

        else:
            # nothing happened, we're done with insn
            assert not new_insn_depends_on

            new_insns.append(insn)

    return kernel.copy(
            instructions=new_insns,
            temporary_variables=new_temporary_variables)

# }}}


# {{{ find boostability of instructions

def find_boostability(kernel):
    logger.debug("%s: boostability" % kernel.name)

    writer_map = kernel.writer_map()

    arg_names = set(arg.name for arg in kernel.args)

    var_names = arg_names | set(six.iterkeys(kernel.temporary_variables))

    dep_map = dict(
            (insn.id, insn.read_dependency_names() & var_names)
            for insn in kernel.instructions)

    non_boostable_vars = set()

    new_insns = []
    for insn in kernel.instructions:
        all_my_var_writers = set()
        for var in dep_map[insn.id]:
            var_writers = writer_map.get(var, set())
            all_my_var_writers |= var_writers

        # {{{ find dependency loops, flag boostability

        while True:
            last_all_my_var_writers = all_my_var_writers

            for writer_insn_id in last_all_my_var_writers:
                for var in dep_map[writer_insn_id]:
                    all_my_var_writers = \
                            all_my_var_writers | writer_map.get(var, set())

            if last_all_my_var_writers == all_my_var_writers:
                break

        # }}}

        boostable = insn.id not in all_my_var_writers

        if not boostable:
            non_boostable_vars.update(
                    var_name for var_name, _ in insn.assignees_and_indices())

        insn = insn.copy(boostable=boostable)

        new_insns.append(insn)

    # {{{ remove boostability from isns that access non-boostable vars

    new2_insns = []
    for insn in new_insns:
        accessed_vars = insn.dependency_names()
        boostable = insn.boostable and not bool(non_boostable_vars & accessed_vars)
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
    # Type inference doesn't handle substitutions. Get them out of the
    # way.

    kernel = infer_unknown_types(kernel, expect_completion=False)

    kernel = add_default_dependencies(kernel)

    # Ordering restrictions:
    #
    # - realize_reduction must happen after type inference because it needs
    #   to be able to determine the types of the reduced expressions.
    #
    # - realize_reduction must happen after default dependencies are added
    #   because it manipulates the depends_on field, which could prevent
    #   defaults from being applied.

    kernel = realize_reduction(kernel)

    # Ordering restriction:
    # add_axes_to_temporaries_for_ilp because reduction accumulators
    # need to be duplicated by this.

    from loopy.transform.ilp import add_axes_to_temporaries_for_ilp_and_vec
    kernel = add_axes_to_temporaries_for_ilp_and_vec(kernel)

    kernel = mark_local_temporaries(kernel)
    kernel = find_boostability(kernel)
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

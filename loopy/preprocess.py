from __future__ import division

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


import pyopencl as cl
import pyopencl.characterize as cl_char
from loopy.diagnostic import (
        LoopyError, LoopyWarning, WriteRaceConditionWarning, warn,
        LoopyAdvisory)

import logging
logger = logging.getLogger(__name__)


# {{{ infer types

def _infer_var_type(kernel, var_name, type_inf_mapper, subst_expander):
    if var_name in kernel.all_params():
        return kernel.index_dtype

    def debug(s):
        logger.debug("%s: %s" % (kernel.name, s))

    dtypes = []

    import loopy as lp

    from loopy.codegen.expression import DependencyTypeInferenceFailure
    for writer_insn_id in kernel.writer_map().get(var_name, []):
        writer_insn = kernel.id_to_insn[writer_insn_id]
        if not isinstance(writer_insn, lp.ExpressionInstruction):
            continue

        expr = subst_expander(writer_insn.expression, insn_id=writer_insn_id)

        try:
            debug("             via expr %s" % expr)
            result = type_inf_mapper(expr)

            debug("             result: %s" % result)

            dtypes.append(result)

        except DependencyTypeInferenceFailure, e:
            debug("             failed: %s" % e)

    if not dtypes:
        return None

    from pytools import is_single_valued
    if not is_single_valued(dtypes):
        raise LoopyError("ambiguous type inference for '%s'"
                % var_name)

    return dtypes[0]


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
    """Infer types on temporaries and argumetns."""

    logger.debug("%s: infer types" % kernel.name)

    def debug(s):
        logger.debug("%s: %s" % (kernel.name, s))

    if kernel.substitutions:
        from warnings import warn as py_warn
        py_warn("type inference called when substitution "
                "rules are still unexpanded, expanding",
                LoopyWarning, stacklevel=2)

        from loopy.subst import expand_subst
        kernel = expand_subst(kernel)

    new_temp_vars = kernel.temporary_variables.copy()
    new_arg_dict = kernel.arg_dict.copy()

    # {{{ fill queue

    # queue contains temporary variables
    queue = []

    import loopy as lp
    for tv in kernel.temporary_variables.itervalues():
        if tv.dtype is lp.auto:
            queue.append(tv)

    for arg in kernel.args:
        if arg.dtype is None:
            queue.append(arg)

    # }}}

    from loopy.codegen.expression import TypeInferenceMapper
    type_inf_mapper = TypeInferenceMapper(kernel,
            _DictUnionView([
                new_temp_vars,
                new_arg_dict
                ]))

    from loopy.symbolic import SubstitutionRuleExpander
    subst_expander = SubstitutionRuleExpander(kernel.substitutions,
            kernel.get_var_name_generator())

    # {{{ work on type inference queue

    from loopy.kernel.data import TemporaryVariable, KernelArgument

    failed_names = set()
    while queue:
        item = queue.pop(0)

        debug("inferring type for %s %s" % (type(item).__name__, item.name))

        result = _infer_var_type(kernel, item.name, type_inf_mapper, subst_expander)

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
                if expect_completion:
                    raise LoopyError(
                            "could not determine type of '%s'" % item.name)
                else:
                    # We're done here.
                    break

            # remember that this item failed
            failed_names.add(item.name)

            queue_names = set(qi.name for qi in queue)

            if queue_names == failed_names:
                # We did what we could...
                print queue_names, failed_names, item.name
                assert not expect_completion
                break

            # can't infer type yet, put back into queue
            queue.append(item)
        else:
            # we've made progress, reset failure markers
            failed_names = set()

    # }}}

    return kernel.copy(
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

    for temp_var in kernel.temporary_variables.itervalues():
        # Only fill out for variables that do not yet know if they're
        # local. (I.e. those generated by implicit temporary generation.)

        if temp_var.is_local is not lp.auto:
            new_temp_vars[temp_var.name] = temp_var
            continue

        my_writers = writers[temp_var.name]

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

        is_local = wants_to_be_local_per_insn[0]
        from pytools import all
        if not all(wtbl == is_local for wtbl in wants_to_be_local_per_insn):
            raise LoopyError("not all instructions agree on whether "
                    "temporary '%s' should be in local memory" % temp_var.name)

        new_temp_vars[temp_var.name] = temp_var.copy(is_local=is_local)

    return kernel.copy(temporary_variables=new_temp_vars)

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

    from loopy.codegen.expression import TypeInferenceMapper
    type_inf_mapper = TypeInferenceMapper(kernel)

    def map_reduction(expr, rec):
        # Only expand one level of reduction at a time, going from outermost to
        # innermost. Otherwise we get the (iname + insn) dependencies wrong.

        from pymbolic import var

        target_var_name = var_name_gen("acc_"+"_".join(expr.inames))
        target_var = var(target_var_name)

        arg_dtype = type_inf_mapper(expr.expr)

        from loopy.kernel.data import ExpressionInstruction, TemporaryVariable

        new_temporary_variables[target_var_name] = TemporaryVariable(
                name=target_var_name,
                shape=(),
                dtype=expr.operation.result_dtype(arg_dtype, expr.inames),
                is_local=False)

        outer_insn_inames = temp_kernel.insn_inames(insn)
        bad_inames = set(expr.inames) & outer_insn_inames
        if bad_inames:
            raise LoopyError("reduction used within loop(s) that it was "
                    "supposed to reduce over: " + ", ".join(bad_inames))

        new_id = temp_kernel.make_unique_instruction_id(
                based_on="%s_%s_init" % (insn.id, "_".join(expr.inames)),
                extra_used_ids=set(i.id for i in generated_insns))

        init_insn = ExpressionInstruction(
                id=new_id,
                assignee=target_var,
                forced_iname_deps=outer_insn_inames - set(expr.inames),
                expression=expr.operation.neutral_element(arg_dtype, expr.inames))

        generated_insns.append(init_insn)

        new_id = temp_kernel.make_unique_instruction_id(
                based_on="%s_%s_update" % (insn.id, "_".join(expr.inames)),
                extra_used_ids=set(i.id for i in generated_insns))

        reduction_insn = ExpressionInstruction(
                id=new_id,
                assignee=target_var,
                expression=expr.operation(
                    arg_dtype, target_var, expr.expr, expr.inames),
                insn_deps=set([init_insn.id]) | insn.insn_deps,
                forced_iname_deps=temp_kernel.insn_inames(insn) | set(expr.inames))

        generated_insns.append(reduction_insn)

        new_insn_insn_deps.add(reduction_insn.id)

        return target_var

    from loopy.symbolic import ReductionCallbackMapper
    cb_mapper = ReductionCallbackMapper(map_reduction)

    insn_queue = kernel.instructions[:]

    temp_kernel = kernel

    import loopy as lp
    while insn_queue:
        new_insn_insn_deps = set()
        generated_insns = []

        insn = insn_queue.pop(0)

        if insn_id_filter is not None and insn.id != insn_id_filter \
                or not isinstance(insn, lp.ExpressionInstruction):
            new_insns.append(insn)
            continue

        # Run reduction expansion.
        new_expression = cb_mapper(insn.expression)

        if generated_insns:
            # An expansion happened, so insert the generated stuff plus
            # ourselves back into the queue.

            insn = insn.copy(
                        expression=new_expression,
                        insn_deps=insn.insn_deps
                            | new_insn_insn_deps,
                        forced_iname_deps=temp_kernel.insn_inames(insn))

            insn_queue = generated_insns + [insn] + insn_queue

            # The reduction expander needs an up-to-date kernel
            # object to find dependencies. Keep temp_kernel up-to-date.

            temp_kernel = kernel.copy(
                    instructions=new_insns + insn_queue,
                    temporary_variables=new_temporary_variables)

        else:
            # nothing happened, we're done with insn
            assert not new_insn_insn_deps

            new_insns.append(insn)

    return kernel.copy(
            instructions=new_insns,
            temporary_variables=new_temporary_variables)

# }}}


# {{{ duplicate private vars for ilp

from loopy.symbolic import IdentityMapper


class ExtraInameIndexInserter(IdentityMapper):
    def __init__(self, var_to_new_inames):
        self.var_to_new_inames = var_to_new_inames

    def map_subscript(self, expr):
        try:
            new_idx = self.var_to_new_inames[expr.aggregate.name]
        except KeyError:
            return IdentityMapper.map_subscript(self, expr)
        else:
            index = expr.index
            if not isinstance(index, tuple):
                index = (index,)
            index = tuple(self.rec(i) for i in index)

            return expr.aggregate[index + new_idx]

    def map_variable(self, expr):
        try:
            new_idx = self.var_to_new_inames[expr.name]
        except KeyError:
            return expr
        else:
            return expr[new_idx]


def duplicate_private_temporaries_for_ilp(kernel):
    logger.debug("%s: duplicate temporaries for ilp" % kernel.name)

    wmap = kernel.writer_map()

    from loopy.kernel.data import IlpBaseTag

    var_to_new_ilp_inames = {}

    # {{{ find variables that need extra indices

    for tv in kernel.temporary_variables.itervalues():
        for writer_insn_id in wmap[tv.name]:
            writer_insn = kernel.id_to_insn[writer_insn_id]
            ilp_inames = frozenset(iname
                    for iname in kernel.insn_inames(writer_insn)
                    if isinstance(kernel.iname_to_tag.get(iname), IlpBaseTag))

            referenced_ilp_inames = (ilp_inames
                    & writer_insn.write_dependency_names())

            new_ilp_inames = ilp_inames - referenced_ilp_inames

            if tv.name in var_to_new_ilp_inames:
                if new_ilp_inames != set(var_to_new_ilp_inames[tv.name]):
                    raise LoopyError("instruction '%s' requires adding "
                            "indices for ILP inames '%s' on var '%s', but previous "
                            "instructions required inames '%s'"
                            % (writer_insn_id, ", ".join(new_ilp_inames),
                                ", ".join(var_to_new_ilp_inames[tv.name])))

                continue

            var_to_new_ilp_inames[tv.name] = set(new_ilp_inames)

    # }}}

    # {{{ find ilp iname lengths

    from loopy.isl_helpers import static_max_of_pw_aff
    from loopy.symbolic import pw_aff_to_expr

    ilp_iname_to_length = {}
    for ilp_inames in var_to_new_ilp_inames.itervalues():
        for iname in ilp_inames:
            if iname in ilp_iname_to_length:
                continue

            bounds = kernel.get_iname_bounds(iname)
            ilp_iname_to_length[iname] = int(pw_aff_to_expr(
                        static_max_of_pw_aff(bounds.size, constants_only=True)))

            assert static_max_of_pw_aff(
                    bounds.lower_bound_pw_aff, constants_only=True).plain_is_zero()

    # }}}

    # {{{ change temporary variables

    new_temp_vars = kernel.temporary_variables.copy()
    for tv_name, inames in var_to_new_ilp_inames.iteritems():
        tv = new_temp_vars[tv_name]
        extra_shape = tuple(ilp_iname_to_length[iname] for iname in inames)

        shape = tv.shape
        if shape is None:
            shape = ()

        new_temp_vars[tv.name] = tv.copy(shape=shape + extra_shape,
                # Forget what you knew about data layout,
                # create from scratch.
                dim_tags=None)

    # }}}

    from pymbolic import var
    eiii = ExtraInameIndexInserter(
            dict((var_name, tuple(var(iname) for iname in inames))
                for var_name, inames in var_to_new_ilp_inames.iteritems()))

    new_insns = [
            insn.with_transformed_expressions(eiii)
            for insn in kernel.instructions]

    return kernel.copy(
        temporary_variables=new_temp_vars,
        instructions=new_insns)

# }}}


# {{{ automatic dependencies, find boostability of instructions

def add_boostability_and_automatic_dependencies(kernel):
    logger.debug("%s: automatic deps, boostability" % kernel.name)

    writer_map = kernel.writer_map()

    arg_names = set(arg.name for arg in kernel.args)

    var_names = arg_names | set(kernel.temporary_variables.iterkeys())

    dep_map = dict(
            (insn.id, insn.read_dependency_names() & var_names)
            for insn in kernel.instructions)

    non_boostable_vars = set()

    new_insns = []
    for insn in kernel.instructions:
        auto_deps = set()

        # {{{ add automatic dependencies

        all_my_var_writers = set()
        for var in dep_map[insn.id]:
            var_writers = writer_map.get(var, set())
            all_my_var_writers |= var_writers

            if not var_writers and var not in arg_names:
                from warnings import warn
                warn("'%s' is read, but never written." % var)

            if len(var_writers) > 1 and not var_writers & set(insn.insn_deps):
                from warnings import warn
                warn("'%s' is written from more than one place, "
                        "but instruction '%s' (which reads this variable) "
                        "does not specify a dependency on any of the writers."
                        % (var, insn.id))

            if len(var_writers) == 1:
                auto_deps.update(var_writers - set([insn.id]))

        # }}}

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

        new_insns.append(
                insn.copy(
                    insn_deps=insn.insn_deps | auto_deps,
                    boostable=boostable))

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

    single_use_inames = set(iname for iname, uc in iname_use_counts.iteritems()
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


# {{{ rank inames by stride

def get_auto_axis_iname_ranking_by_stride(kernel, insn):
    from loopy.kernel.data import ImageArg, ValueArg

    approximate_arg_values = {}
    for arg in kernel.args:
        if isinstance(arg, ValueArg):
            if arg.approximately is not None:
                approximate_arg_values[arg.name] = arg.approximately
            else:
                raise LoopyError("No approximate arg value specified for '%s'"
                        % arg.name)

    # {{{ find all array accesses in insn

    from loopy.symbolic import ArrayAccessFinder
    ary_acc_exprs = list(ArrayAccessFinder()(insn.expression))

    from pymbolic.primitives import Subscript

    if isinstance(insn.assignee, Subscript):
        ary_acc_exprs.append(insn.assignee)

    # }}}

    # {{{ filter array accesses to only the global ones

    global_ary_acc_exprs = []

    for aae in ary_acc_exprs:
        ary_name = aae.aggregate.name
        arg = kernel.arg_dict.get(ary_name)
        if arg is None:
            continue

        if isinstance(arg, ImageArg):
            continue

        global_ary_acc_exprs.append(aae)

    # }}}

    # {{{ figure out automatic-axis inames

    from loopy.kernel.data import AutoLocalIndexTagBase
    auto_axis_inames = set(
            iname
            for iname in kernel.insn_inames(insn)
            if isinstance(kernel.iname_to_tag.get(iname),
                AutoLocalIndexTagBase))

    # }}}

    # {{{ figure out which iname should get mapped to local axis 0

    # maps inames to "aggregate stride"
    aggregate_strides = {}

    from loopy.symbolic import CoefficientCollector
    from pymbolic.primitives import Variable

    for aae in global_ary_acc_exprs:
        index_expr = aae.index
        if not isinstance(index_expr, tuple):
            index_expr = (index_expr,)

        ary_name = aae.aggregate.name
        arg = kernel.arg_dict.get(ary_name)

        if arg.dim_tags is None:
            from warnings import warn
            warn("Strides for '%s' are not known. Local axis assignment "
                    "is likely suboptimal." % arg.name)
            ary_strides = [1] * len(index_expr)
        else:
            ary_strides = []
            from loopy.kernel.array import FixedStrideArrayDimTag
            for dim_tag in arg.dim_tags:
                if isinstance(dim_tag, FixedStrideArrayDimTag):
                    ary_strides.append(dim_tag.stride)

        # {{{ construct iname_to_stride_expr

        iname_to_stride_expr = {}
        for iexpr_i, stride in zip(index_expr, ary_strides):
            if stride is None:
                continue
            coeffs = CoefficientCollector()(iexpr_i)
            for var, coeff in coeffs.iteritems():
                assert isinstance(var, Variable)
                if var.name in auto_axis_inames:  # excludes '1', i.e.  the constant
                    new_stride = coeff*stride
                    old_stride = iname_to_stride_expr.get(var.name, None)
                    if old_stride is None or new_stride < old_stride:
                        iname_to_stride_expr[var.name] = new_stride

        # }}}

        from pymbolic import evaluate
        for iname, stride_expr in iname_to_stride_expr.iteritems():
            stride = evaluate(stride_expr, approximate_arg_values)
            aggregate_strides[iname] = aggregate_strides.get(iname, 0) + stride

    if aggregate_strides:
        import sys
        return sorted((iname for iname in kernel.insn_inames(insn)),
                key=lambda iname: aggregate_strides.get(iname, sys.maxint))
    else:
        return None

    # }}}

# }}}


# {{{ assign automatic axes

def assign_automatic_axes(kernel, axis=0, local_size=None):
    logger.debug("%s: assign automatic axes" % kernel.name)

    from loopy.kernel.data import (AutoLocalIndexTagBase, LocalIndexTag)

    # Realize that at this point in time, axis lengths are already
    # fixed. So we compute them once and pass them to our recursive
    # copies.

    if local_size is None:
        _, local_size = kernel.get_grid_sizes_as_exprs(
                ignore_auto=True)

    # {{{ axis assignment helper function

    def assign_axis(recursion_axis, iname, axis=None):
        """Assign iname to local axis *axis* and start over by calling
        the surrounding function assign_automatic_axes.

        If *axis* is None, find a suitable axis automatically.
        """
        desired_length = kernel.get_constant_iname_length(iname)

        if axis is None:
            # {{{ find a suitable axis

            shorter_possible_axes = []
            test_axis = 0
            while True:
                if test_axis >= len(local_size):
                    break
                if test_axis in assigned_local_axes:
                    test_axis += 1
                    continue

                if local_size[test_axis] < desired_length:
                    shorter_possible_axes.append(test_axis)
                    test_axis += 1
                    continue
                else:
                    axis = test_axis
                    break

            # The loop above will find an unassigned local axis
            # that has enough 'room' for the iname. In the same traversal,
            # it also finds theoretically assignable axes that are shorter,
            # in the variable shorter_possible_axes.

            if axis is None and shorter_possible_axes:
                # sort as longest first
                shorter_possible_axes.sort(key=lambda ax: local_size[ax])
                axis = shorter_possible_axes[0]

            # }}}

        if axis is None:
            new_tag = None
        else:
            new_tag = LocalIndexTag(axis)
            if desired_length > local_size[axis]:
                from loopy import split_iname

                # Don't be tempted to switch the outer tag to unroll--this may
                # generate tons of code on some examples.

                return assign_automatic_axes(
                        split_iname(kernel, iname, inner_length=local_size[axis],
                            outer_tag=None, inner_tag=new_tag,
                            do_tagged_check=False),
                        axis=recursion_axis, local_size=local_size)

        if not isinstance(kernel.iname_to_tag.get(iname), AutoLocalIndexTagBase):
            raise LoopyError("trying to reassign '%s'" % iname)

        new_iname_to_tag = kernel.iname_to_tag.copy()
        new_iname_to_tag[iname] = new_tag
        return assign_automatic_axes(kernel.copy(iname_to_tag=new_iname_to_tag),
                axis=recursion_axis, local_size=local_size)

    # }}}

    # {{{ main assignment loop

    # assignment proceeds in one phase per axis, each time assigning the
    # smallest-stride available iname to the current axis

    import loopy as lp

    for insn in kernel.instructions:
        if not isinstance(insn, lp.ExpressionInstruction):
            continue

        auto_axis_inames = [
                iname
                for iname in kernel.insn_inames(insn)
                if isinstance(kernel.iname_to_tag.get(iname),
                    AutoLocalIndexTagBase)]

        if not auto_axis_inames:
            continue

        assigned_local_axes = set()

        for iname in kernel.insn_inames(insn):
            tag = kernel.iname_to_tag.get(iname)
            if isinstance(tag, LocalIndexTag):
                assigned_local_axes.add(tag.axis)

        if axis < len(local_size):
            # "valid" pass: try to assign a given axis

            if axis not in assigned_local_axes:
                iname_ranking = get_auto_axis_iname_ranking_by_stride(kernel, insn)
                if iname_ranking is not None:
                    for iname in iname_ranking:
                        prev_tag = kernel.iname_to_tag.get(iname)
                        if isinstance(prev_tag, AutoLocalIndexTagBase):
                            return assign_axis(axis, iname, axis)

        else:
            # "invalid" pass: There are still unassigned axis after the
            #  numbered "valid" passes--assign the remainder by length.

            # assign longest auto axis inames first
            auto_axis_inames.sort(key=kernel.get_constant_iname_length, reverse=True)

            if auto_axis_inames:
                return assign_axis(axis, auto_axis_inames.pop())

    # }}}

    # We've seen all instructions and not punted to recursion/restart because
    # of a new axis assignment.

    if axis >= len(local_size):
        return kernel
    else:
        return assign_automatic_axes(kernel, axis=axis+1,
                local_size=local_size)

# }}}


# {{{ temp storage adjust for bank conflict

def adjust_local_temp_var_storage(kernel):
    logger.debug("%s: adjust temp var storage" % kernel.name)

    new_temp_vars = {}

    lmem_size = cl_char.usable_local_mem_size(kernel.device)
    for temp_var in kernel.temporary_variables.itervalues():
        if not temp_var.is_local:
            new_temp_vars[temp_var.name] = \
                    temp_var.copy(storage_shape=temp_var.shape)
            continue

        other_loctemp_nbytes = [
                tv.nbytes
                for tv in kernel.temporary_variables.itervalues()
                if tv.is_local and tv.name != temp_var.name]

        storage_shape = temp_var.storage_shape

        if storage_shape is None:
            storage_shape = temp_var.shape

        storage_shape = list(storage_shape)

        # sizes of all dims except the last one, which we may change
        # below to avoid bank conflicts
        from pytools import product

        if kernel.device.local_mem_type == cl.device_local_mem_type.GLOBAL:
            # FIXME: could try to avoid cache associativity disasters
            new_storage_shape = storage_shape

        elif kernel.device.local_mem_type == cl.device_local_mem_type.LOCAL:
            min_mult = cl_char.local_memory_bank_count(kernel.device)
            good_incr = None
            new_storage_shape = storage_shape
            min_why_not = None

            for increment in range(storage_shape[-1]//2):

                test_storage_shape = storage_shape[:]
                test_storage_shape[-1] = test_storage_shape[-1] + increment
                new_mult, why_not = cl_char.why_not_local_access_conflict_free(
                        kernel.device, temp_var.dtype.itemsize,
                        temp_var.shape, test_storage_shape)

                # will choose smallest increment 'automatically'
                if new_mult < min_mult:
                    new_lmem_use = (sum(other_loctemp_nbytes)
                            + temp_var.dtype.itemsize*product(test_storage_shape))
                    if new_lmem_use < lmem_size:
                        new_storage_shape = test_storage_shape
                        min_mult = new_mult
                        min_why_not = why_not
                        good_incr = increment

            if min_mult != 1:
                from warnings import warn
                from loopy.diagnostic import LoopyAdvisory
                warn("could not find a conflict-free mem layout "
                        "for local variable '%s' "
                        "(currently: %dx conflict, increment: %s, reason: %s)"
                        % (temp_var.name, min_mult, good_incr, min_why_not),
                        LoopyAdvisory)
        else:
            from warnings import warn
            warn("unknown type of local memory")

            new_storage_shape = storage_shape

        new_temp_vars[temp_var.name] = temp_var.copy(storage_shape=new_storage_shape)

    return kernel.copy(temporary_variables=new_temp_vars)

# }}}


def preprocess_kernel(kernel):
    logger.info("%s: preprocess start" % kernel.name)

    from loopy.subst import expand_subst
    kernel = expand_subst(kernel)

    # Ordering restriction:
    # Type inference doesn't handle substitutions. Get them out of the
    # way.

    kernel = infer_unknown_types(kernel, expect_completion=False)

    # Ordering restriction:
    # realize_reduction must happen after type inference because it needs
    # to be able to determine the types of the reduced expressions.

    kernel = realize_reduction(kernel)

    # Ordering restriction:
    # duplicate_private_temporaries_for_ilp because reduction accumulators
    # need to be duplicated by this.

    kernel = duplicate_private_temporaries_for_ilp(kernel)
    kernel = mark_local_temporaries(kernel)
    kernel = assign_automatic_axes(kernel)
    kernel = add_boostability_and_automatic_dependencies(kernel)
    kernel = limit_boostability(kernel)
    kernel = adjust_local_temp_var_storage(kernel)

    logger.info("%s: preprocess done" % kernel.name)

    return kernel




# vim: foldmethod=marker

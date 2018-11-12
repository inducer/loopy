# coding=utf-8
"""Operations on the kernel object."""

from __future__ import division, absolute_import, print_function

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

import sys

import six
from six.moves import intern

import numpy as np
import islpy as isl
from islpy import dim_type
from loopy.diagnostic import LoopyError, warn_with_kernel
from pytools import memoize_on_first_arg
from loopy.tools import natsorted

import logging
logger = logging.getLogger(__name__)


# {{{ add and infer argument dtypes

def add_dtypes(knl, dtype_dict):
    """Specify remaining unspecified argument/temporary variable types.

    :arg dtype_dict: a mapping from variable names to :class:`numpy.dtype`
        instances
    """
    dtype_dict_remainder, new_args, new_temp_vars = _add_dtypes(knl, dtype_dict)

    if dtype_dict_remainder:
        raise RuntimeError("unused argument dtypes: %s"
                % ", ".join(dtype_dict_remainder))

    return knl.copy(args=new_args, temporary_variables=new_temp_vars)


def _add_dtypes_overdetermined(knl, dtype_dict):
    dtype_dict_remainder, new_args, new_temp_vars = _add_dtypes(knl, dtype_dict)
    # do not throw error for unused args
    return knl.copy(args=new_args, temporary_variables=new_temp_vars)


def _add_dtypes(knl, dtype_dict):
    dtype_dict = dtype_dict.copy()
    new_args = []

    from loopy.types import to_loopy_type
    for arg in knl.args:
        new_dtype = dtype_dict.pop(arg.name, None)
        if new_dtype is not None:
            new_dtype = to_loopy_type(new_dtype, target=knl.target)
            if arg.dtype is not None and arg.dtype != new_dtype:
                raise RuntimeError(
                        "argument '%s' already has a different dtype "
                        "(existing: %s, new: %s)"
                        % (arg.name, arg.dtype, new_dtype))
            arg = arg.copy(dtype=new_dtype)

        new_args.append(arg)

    new_temp_vars = knl.temporary_variables.copy()

    import loopy as lp
    for tv_name in knl.temporary_variables:
        new_dtype = dtype_dict.pop(tv_name, None)
        if new_dtype is not None:
            new_dtype = np.dtype(new_dtype)
            tv = new_temp_vars[tv_name]
            if (tv.dtype is not None and tv.dtype is not lp.auto) \
                    and tv.dtype != new_dtype:
                raise RuntimeError(
                        "temporary variable '%s' already has a different dtype "
                        "(existing: %s, new: %s)"
                        % (tv_name, tv.dtype, new_dtype))

            new_temp_vars[tv_name] = tv.copy(dtype=new_dtype)

    return dtype_dict, new_args, new_temp_vars


def get_arguments_with_incomplete_dtype(knl):
    return [arg.name for arg in knl.args
            if arg.dtype is None]


def add_and_infer_dtypes(knl, dtype_dict, expect_completion=False):
    processed_dtype_dict = {}

    for k, v in six.iteritems(dtype_dict):
        for subkey in k.split(","):
            subkey = subkey.strip()
            if subkey:
                processed_dtype_dict[subkey] = v

    knl = add_dtypes(knl, processed_dtype_dict)

    from loopy.type_inference import infer_unknown_types
    return infer_unknown_types(knl, expect_completion=expect_completion)


def _add_and_infer_dtypes_overdetermined(knl, dtype_dict):
    knl = _add_dtypes_overdetermined(knl, dtype_dict)

    from loopy.type_inference import infer_unknown_types
    return infer_unknown_types(knl, expect_completion=True)

# }}}


# {{{ find_all_insn_inames fixed point iteration (deprecated)

def guess_iname_deps_based_on_var_use(kernel, insn, insn_id_to_inames=None):
    # For all variables that insn depends on, find the intersection
    # of iname deps of all writers, and add those to insn's
    # dependencies.

    result = frozenset()

    writer_map = kernel.writer_map()

    for tv_name in (insn.read_dependency_names() & kernel.get_written_variables()):
        tv_implicit_inames = None

        for writer_id in writer_map[tv_name]:
            writer_insn = kernel.id_to_insn[writer_id]
            if insn_id_to_inames is None:
                writer_inames = writer_insn.within_inames
            else:
                writer_inames = insn_id_to_inames[writer_id]

            writer_implicit_inames = (
                    writer_inames
                    - (writer_insn.write_dependency_names() & kernel.all_inames()))
            if tv_implicit_inames is None:
                tv_implicit_inames = writer_implicit_inames
            else:
                tv_implicit_inames = (tv_implicit_inames
                        & writer_implicit_inames)

        if tv_implicit_inames is not None:
            result = result | tv_implicit_inames

    return result - insn.reduction_inames()


def find_all_insn_inames(kernel):
    logger.debug("%s: find_all_insn_inames: start" % kernel.name)

    writer_map = kernel.writer_map()

    insn_id_to_inames = {}
    insn_assignee_inames = {}

    all_read_deps = {}
    all_write_deps = {}

    from loopy.transform.subst import expand_subst
    kernel = expand_subst(kernel)

    for insn in kernel.instructions:
        all_read_deps[insn.id] = read_deps = insn.read_dependency_names()
        all_write_deps[insn.id] = write_deps = insn.write_dependency_names()
        deps = read_deps | write_deps

        if insn.within_inames_is_final:
            iname_deps = insn.within_inames
        else:
            iname_deps = (
                    deps & kernel.all_inames()
                    | insn.within_inames)

        assert isinstance(read_deps, frozenset), type(insn)
        assert isinstance(write_deps, frozenset), type(insn)
        assert isinstance(iname_deps, frozenset), type(insn)

        logger.debug("%s: find_all_insn_inames: %s (init): %s - "
                "read deps: %s - write deps: %s" % (
                    kernel.name, insn.id, ", ".join(sorted(iname_deps)),
                    ", ".join(sorted(read_deps)), ", ".join(sorted(write_deps)),
                    ))

        insn_id_to_inames[insn.id] = iname_deps
        insn_assignee_inames[insn.id] = write_deps & kernel.all_inames()

    # fixed point iteration until all iname dep sets have converged

    # Why is fixed point iteration necessary here? Consider the following
    # scenario:
    #
    # z = expr(iname)
    # y = expr(z)
    # x = expr(y)
    #
    # x clearly has a dependency on iname, but this is not found until that
    # dependency has propagated all the way up. Doing this recursively is
    # not guaranteed to terminate because of circular dependencies.

    while True:
        did_something = False
        for insn in kernel.instructions:

            if insn.within_inames_is_final:
                continue

            # {{{ depdency-based propagation

            inames_old = insn_id_to_inames[insn.id]
            inames_new = inames_old | guess_iname_deps_based_on_var_use(
                    kernel, insn, insn_id_to_inames)

            insn_id_to_inames[insn.id] = inames_new

            if inames_new != inames_old:
                did_something = True

                warn_with_kernel(kernel, "inferred_iname",
                        "The iname(s) '%s' on instruction '%s' "
                        "was/were automatically added. "
                        "This is deprecated. Please add the iname "
                        "to the instruction "
                        "explicitly, e.g. by adding 'for' loops"
                        % (", ".join(inames_new-inames_old), insn.id))

            # }}}

            # {{{ domain-based propagation

            inames_old = insn_id_to_inames[insn.id]
            inames_new = set(insn_id_to_inames[insn.id])

            for iname in inames_old:
                home_domain = kernel.domains[kernel.get_home_domain_index(iname)]

                for par in home_domain.get_var_names(dim_type.param):
                    # Add all inames occurring in parameters of domains that my
                    # current inames refer to.

                    if par in kernel.all_inames():
                        inames_new.add(intern(par))

                    # If something writes the bounds of a loop in which I'm
                    # sitting, I had better be in the inames that the writer is
                    # in.

                    if par in kernel.temporary_variables:
                        for writer_id in writer_map.get(par, []):
                            inames_new.update(insn_id_to_inames[writer_id])

            if inames_new != inames_old:
                did_something = True
                insn_id_to_inames[insn.id] = frozenset(inames_new)

                warn_with_kernel(kernel, "inferred_iname",
                        "The iname(s) '%s' on instruction '%s' was "
                        "automatically added. "
                        "This is deprecated. Please add the iname "
                        "to the instruction "
                        "explicitly, e.g. by adding 'for' loops"
                        % (", ".join(inames_new-inames_old), insn.id))

            # }}}

        if not did_something:
            break

    logger.debug("%s: find_all_insn_inames: done" % kernel.name)

    for v in six.itervalues(insn_id_to_inames):
        assert isinstance(v, frozenset)

    return insn_id_to_inames

# }}}


# {{{ set operation cache

class SetOperationCacheManager:
    def __init__(self):
        # mapping: set hash -> [(set, op, args, result)]
        self.cache = {}

    def op(self, set, op_name, op, args):
        hashval = hash(set)
        bucket = self.cache.setdefault(hashval, [])

        for bkt_set, bkt_op, bkt_args, result in bucket:
            if set.plain_is_equal(bkt_set) and op == bkt_op and args == bkt_args:
                return result

        #print op, set.get_dim_name(dim_type.set, args[0])
        result = op(set, *args)
        bucket.append((set, op_name, args, result))
        return result

    def dim_min(self, set, *args):
        if set.plain_is_empty():
            raise LoopyError("domain '%s' is empty" % set)

        from loopy.isl_helpers import dim_min_with_elimination
        return self.op(set, "dim_min", dim_min_with_elimination, args)

    def dim_max(self, set, *args):
        if set.plain_is_empty():
            raise LoopyError("domain '%s' is empty" % set)

        from loopy.isl_helpers import dim_max_with_elimination
        return self.op(set, "dim_max", dim_max_with_elimination, args)

    def base_index_and_length(self, set, iname, context=None,
            n_allowed_params_in_length=None):
        """
        :arg n_allowed_params_in_length: Simplifies the 'length'
            argument so that only the first that many params
            (in the domain of *set*) occur.
        """
        if not isinstance(iname, int):
            iname_to_dim = set.space.get_var_dict()
            idx = iname_to_dim[iname][1]
        else:
            idx = iname

        lower_bound_pw_aff = self.dim_min(set, idx)
        upper_bound_pw_aff = self.dim_max(set, idx)

        from loopy.diagnostic import StaticValueFindingError
        from loopy.isl_helpers import (
                static_max_of_pw_aff,
                static_min_of_pw_aff,
                static_value_of_pw_aff,
                find_max_of_pwaff_with_params)
        from loopy.symbolic import pw_aff_to_expr

        # {{{ first: try to find static lower bound value

        try:
            base_index_aff = static_value_of_pw_aff(
                    lower_bound_pw_aff, constants_only=False,
                    context=context)
        except StaticValueFindingError:
            base_index_aff = None

        if base_index_aff is not None:
            base_index = pw_aff_to_expr(base_index_aff)

            length = find_max_of_pwaff_with_params(
                    upper_bound_pw_aff - base_index_aff + 1,
                    n_allowed_params_in_length)
            length = pw_aff_to_expr(static_max_of_pw_aff(
                    length, constants_only=False,
                    context=context))

            return base_index, length

        # }}}

        # {{{ if that didn't work, try finding a lower bound

        base_index_aff = static_min_of_pw_aff(
                lower_bound_pw_aff, constants_only=False,
                context=context)

        base_index = pw_aff_to_expr(base_index_aff)

        length = find_max_of_pwaff_with_params(
                upper_bound_pw_aff - base_index_aff + 1,
                n_allowed_params_in_length)
        length = pw_aff_to_expr(static_max_of_pw_aff(
                length, constants_only=False,
                context=context))

        return base_index, length

        # }}}

# }}}


# {{{ domain change helper

class DomainChanger:
    """Helps change the domain responsible for *inames* within a kernel.

    .. note: Does not perform an in-place change!
    """

    def __init__(self, kernel, inames):
        """
        :arg inames: a non-mutable iterable
        """

        self.kernel = kernel
        if inames:
            ldi = kernel.get_leaf_domain_indices(inames)
            if len(ldi) > 1:
                raise RuntimeError("Inames '%s' require more than one leaf "
                        "domain, which makes the domain change that is part "
                        "of your current operation ambiguous." % ", ".join(inames))

            self.leaf_domain_index, = ldi
            self.domain = kernel.domains[self.leaf_domain_index]

        else:
            self.domain = kernel.combine_domains(())
            self.leaf_domain_index = None

    def get_original_domain(self):
        return self.kernel.domains[self.leaf_domain_index]

    def get_domains_with(self, replacement):
        result = self.kernel.domains[:]
        if self.leaf_domain_index is not None:
            result[self.leaf_domain_index] = replacement
        else:
            result.append(replacement)

        return result

    def get_kernel_with(self, replacement):
        return self.kernel.copy(
                domains=self.get_domains_with(replacement),

                # Changing the domain might look like it wants to change grid
                # sizes. Not true.
                # (Relevant for 'slab decomposition')
                overridden_get_grid_sizes_for_insn_ids=(
                    self.kernel.get_grid_sizes_for_insn_ids))

# }}}


# {{{ graphviz / dot export

def get_dot_dependency_graph(kernel, iname_cluster=True, use_insn_id=False):
    """Return a string in the `dot <http://graphviz.org/>`_ language depicting
    dependencies among kernel instructions.
    """

    # make sure all automatically added stuff shows up
    from loopy.kernel.creation import apply_single_writer_depencency_heuristic
    kernel = apply_single_writer_depencency_heuristic(kernel, warn_if_used=False)

    if iname_cluster and not kernel.schedule:
        try:
            from loopy.schedule import get_one_scheduled_kernel
            kernel = get_one_scheduled_kernel(kernel)
        except RuntimeError as e:
            iname_cluster = False
            from warnings import warn
            warn("error encountered during scheduling for dep graph -- "
                    "cannot perform iname clustering: %s(%s)"
                    % (type(e).__name__, e))

    dep_graph = {}
    lines = []

    from loopy.kernel.data import MultiAssignmentBase, CInstruction

    for insn in kernel.instructions:
        if isinstance(insn, MultiAssignmentBase):
            op = "%s <- %s" % (insn.assignees, insn.expression)
            if len(op) > 200:
                op = op[:200] + "..."

        elif isinstance(insn, CInstruction):
            op = "<C instruction %s>" % insn.id
        else:
            op = "<instruction %s>" % insn.id

        if use_insn_id:
            insn_label = insn.id
            tooltip = op
        else:
            insn_label = op
            tooltip = insn.id

        lines.append("\"%s\" [label=\"%s\",shape=\"box\",tooltip=\"%s\"];"
                % (
                    insn.id,
                    repr(insn_label)[1:-1],
                    repr(tooltip)[1:-1],
                    ))
        for dep in insn.depends_on:
            dep_graph.setdefault(insn.id, set()).add(dep)

    # {{{ O(n^3) transitive reduction

    # first, compute transitive closure by fixed point iteration
    while True:
        changed_something = False

        for insn_1 in dep_graph:
            for insn_2 in dep_graph.get(insn_1, set()).copy():
                for insn_3 in dep_graph.get(insn_2, set()).copy():
                    if insn_3 not in dep_graph.get(insn_1, set()):
                        changed_something = True
                        dep_graph[insn_1].add(insn_3)

        if not changed_something:
            break

    for insn_1 in dep_graph:
        for insn_2 in dep_graph.get(insn_1, set()).copy():
            for insn_3 in dep_graph.get(insn_2, set()).copy():
                if insn_3 in dep_graph.get(insn_1, set()):
                    dep_graph[insn_1].remove(insn_3)

    # }}}

    for insn_1 in dep_graph:
        for insn_2 in dep_graph.get(insn_1, set()):
            lines.append("%s -> %s" % (insn_2, insn_1))

    if iname_cluster:
        from loopy.schedule import (
                EnterLoop, LeaveLoop, RunInstruction, Barrier,
                CallKernel, ReturnFromKernel)

        for sched_item in kernel.schedule:
            if isinstance(sched_item, EnterLoop):
                lines.append("subgraph cluster_%s { label=\"%s\""
                        % (sched_item.iname, sched_item.iname))
            elif isinstance(sched_item, LeaveLoop):
                lines.append("}")
            elif isinstance(sched_item, RunInstruction):
                lines.append(sched_item.insn_id)
            elif isinstance(sched_item, (CallKernel, ReturnFromKernel, Barrier)):
                pass
            else:
                raise LoopyError("schedule item not unterstood: %r" % sched_item)

    return "digraph %s {\n%s\n}" % (
            kernel.name,
            "\n".join(lines)
            )


def show_dependency_graph(*args, **kwargs):
    """Show the dependency graph generated by :func:`get_dot_dependency_graph`
    in a browser. Accepts the same arguments as that function.
    """

    dot = get_dot_dependency_graph(*args, **kwargs)

    from tempfile import mkdtemp
    temp_dir = mkdtemp(prefix="tmp_loopy_dot")

    dot_file_name = "loopy.dot"

    from os.path import join
    with open(join(temp_dir, dot_file_name), "w") as dotf:
        dotf.write(dot)

    svg_file_name = "loopy.svg"
    from subprocess import check_call
    check_call(["dot", "-Tsvg", "-o", svg_file_name, dot_file_name],
            cwd=temp_dir)

    full_svg_file_name = join(temp_dir, svg_file_name)
    logger.info("show_dot_dependency_graph: svg written to '%s'"
            % full_svg_file_name)

    from webbrowser import open as browser_open
    browser_open("file://" + full_svg_file_name)

# }}}


# {{{ is domain dependent on inames

def is_domain_dependent_on_inames(kernel, domain_index, inames):
    dom = kernel.domains[domain_index]
    dom_parameters = set(dom.get_var_names(dim_type.param))

    # {{{ check for parenthood by loop bound iname

    if inames & dom_parameters:
        return True

    # }}}

    # {{{ check for parenthood by written variable

    for par in dom_parameters:
        if par in kernel.temporary_variables:
            writer_insns = kernel.writer_map()[par]

            if len(writer_insns) > 1:
                raise RuntimeError("loop bound '%s' "
                        "may only be written to once" % par)

            writer_insn, = writer_insns
            writer_inames = kernel.insn_inames(writer_insn)

            if writer_inames & inames:
                return True

    # }}}

    return False

# }}}


# {{{ assign automatic axes

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

    for assignee in insn.assignees:
        if isinstance(assignee, Subscript):
            ary_acc_exprs.append(assignee)

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
        iname for iname in kernel.insn_inames(insn)
        if kernel.iname_tags_of_type(iname, AutoLocalIndexTagBase))

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
            for var, coeff in six.iteritems(coeffs):
                if (isinstance(var, Variable)
                        and var.name in auto_axis_inames):
                    # excludes '1', i.e.  the constant
                    new_stride = coeff*stride
                    old_stride = iname_to_stride_expr.get(var.name, None)
                    if old_stride is None or new_stride < old_stride:
                        iname_to_stride_expr[var.name] = new_stride

        # }}}

        from pymbolic import evaluate
        for iname, stride_expr in six.iteritems(iname_to_stride_expr):
            stride = evaluate(stride_expr, approximate_arg_values)
            aggregate_strides[iname] = aggregate_strides.get(iname, 0) + stride

    if aggregate_strides:
        very_large_stride = int(np.iinfo(np.int32).max)

        return sorted((iname for iname in kernel.insn_inames(insn)),
                key=lambda iname: (
                    aggregate_strides.get(iname, very_large_stride),
                    iname))
    else:
        return None

    # }}}

# }}}


def assign_automatic_axes(kernel, axis=0, local_size=None):
    logger.debug("%s: assign automatic axes" % kernel.name)
    # TODO: do the tag removal rigorously, might be easier after switching
    # to set() from tuple()

    from loopy.kernel.data import (AutoLocalIndexTagBase, LocalIndexTag,
                                   filter_iname_tags_by_type)

    # Realize that at this point in time, axis lengths are already
    # fixed. So we compute them once and pass them to our recursive
    # copies.

    if local_size is None:
        _, local_size = kernel.get_grid_size_upper_bounds_as_exprs(
                ignore_auto=True)

    # {{{ axis assignment helper function

    def assign_axis(recursion_axis, iname, axis=None):
        """Assign iname to local axis *axis* and start over by calling
        the surrounding function assign_automatic_axes.

        If *axis* is None, find a suitable axis automatically.
        """
        try:
            with isl.SuppressedWarnings(kernel.isl_context):
                desired_length = kernel.get_constant_iname_length(iname)
        except isl.Error:
            # Likely unbounded, automatic assignment is not
            # going to happen for this iname.
            new_iname_to_tags = kernel.iname_to_tags.copy()
            new_tags = new_iname_to_tags.get(iname, frozenset())
            new_tags = frozenset(tag for tag in new_tags
                    if not isinstance(tag, AutoLocalIndexTagBase))

            if new_tags:
                new_iname_to_tags[iname] = new_tags
            else:
                del new_iname_to_tags[iname]

            return assign_automatic_axes(
                    kernel.copy(iname_to_tags=new_iname_to_tags),
                    axis=recursion_axis)

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
                from loopy import split_iname, untag_inames

                # Don't be tempted to switch the outer tag to unroll--this may
                # generate tons of code on some examples.

                return assign_automatic_axes(
                        split_iname(
                            untag_inames(kernel, iname, AutoLocalIndexTagBase),
                            iname, inner_length=local_size[axis],
                            outer_tag=None, inner_tag=new_tag,
                            do_tagged_check=False),
                        axis=recursion_axis, local_size=local_size)

        if not kernel.iname_tags_of_type(iname, AutoLocalIndexTagBase):
            raise LoopyError("trying to reassign '%s'" % iname)

        if new_tag:
            new_tag_set = frozenset([new_tag])
        else:
            new_tag_set = frozenset()
        new_iname_to_tags = kernel.iname_to_tags.copy()
        new_tags = (
                frozenset(tag for tag in new_iname_to_tags.get(iname, frozenset())
                    if not isinstance(tag, AutoLocalIndexTagBase))
                | new_tag_set)

        if new_tags:
            new_iname_to_tags[iname] = new_tags
        else:
            del new_iname_to_tags[iname]

        return assign_automatic_axes(kernel.copy(iname_to_tags=new_iname_to_tags),
                axis=recursion_axis, local_size=local_size)

    # }}}

    # {{{ main assignment loop

    # assignment proceeds in one phase per axis, each time assigning the
    # smallest-stride available iname to the current axis

    import loopy as lp

    for insn in kernel.instructions:
        if not isinstance(insn, lp.MultiAssignmentBase):
            continue

        auto_axis_inames = [
            iname for iname in kernel.insn_inames(insn)
            if kernel.iname_tags_of_type(iname, AutoLocalIndexTagBase)]

        if not auto_axis_inames:
            continue

        assigned_local_axes = set()

        for iname in kernel.insn_inames(insn):
            tags = kernel.iname_tags_of_type(iname, LocalIndexTag, max_num=1)
            if tags:
                tag, = tags
                assigned_local_axes.add(tag.axis)

        if axis < len(local_size):
            # "valid" pass: try to assign a given axis

            if axis not in assigned_local_axes:
                iname_ranking = get_auto_axis_iname_ranking_by_stride(kernel, insn)
                if iname_ranking is not None:
                    for iname in iname_ranking:
                        prev_tags = kernel.iname_tags(iname)
                        if filter_iname_tags_by_type(
                                prev_tags, AutoLocalIndexTagBase):
                            return assign_axis(axis, iname, axis)

        else:
            # "invalid" pass: There are still unassigned axis after the
            #  numbered "valid" passes--assign the remainder by length.

            def get_iname_length(iname):
                try:
                    with isl.SuppressedWarnings(kernel.isl_context):
                        return kernel.get_constant_iname_length(iname)
                except isl.Error:
                    return -1
            # assign longest auto axis inames first
            auto_axis_inames.sort(
                            key=get_iname_length,
                            reverse=True)

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


# {{{ array modifier

class ArrayChanger(object):
    def __init__(self, kernel, array_name):
        self.kernel = kernel
        self.array_name = array_name

    def get(self):
        ary_name = self.array_name
        if ary_name in self.kernel.temporary_variables:
            result = self.kernel.temporary_variables[ary_name]
        elif ary_name in self.kernel.arg_dict:
            result = self.kernel.arg_dict[ary_name]
        else:
            raise NameError("array '%s' was not found" % ary_name)

        from loopy.kernel.array import ArrayBase
        if not isinstance(result, ArrayBase):
            raise LoopyError("variable '%s' is not an array" % ary_name)

        return result

    def with_changed_array(self, new_array):
        knl = self.kernel
        ary_name = self.array_name

        if ary_name in knl.temporary_variables:
            new_tv = knl.temporary_variables.copy()
            new_tv[ary_name] = new_array
            return knl.copy(temporary_variables=new_tv)

        elif ary_name in knl.arg_dict:
            new_args = []
            for arg in knl.args:
                if arg.name == ary_name:
                    new_args.append(new_array)
                else:
                    new_args.append(arg)

            return knl.copy(args=new_args)

        else:
            raise NameError("array '%s' was not found" % ary_name)

# }}}


# {{{ guess_var_shape

def guess_var_shape(kernel, var_name):
    from loopy.symbolic import SubstitutionRuleExpander, AccessRangeMapper

    armap = AccessRangeMapper(kernel, var_name)

    submap = SubstitutionRuleExpander(kernel.substitutions)

    def run_through_armap(expr):
        armap(submap(expr), kernel.insn_inames(insn))
        return expr

    try:
        for insn in kernel.instructions:
            insn.with_transformed_expressions(run_through_armap)
    except TypeError as e:
        from traceback import print_exc
        print_exc()

        raise LoopyError(
                "Failed to (automatically, as requested) find "
                "shape/strides for variable '%s'. "
                "Specifying the shape manually should get rid of this. "
                "The following error occurred: %s"
                % (var_name, str(e)))

    if armap.access_range is None:
        if armap.bad_subscripts:
            from loopy.symbolic import LinearSubscript
            if any(isinstance(sub, LinearSubscript)
                    for sub in armap.bad_subscripts):
                raise LoopyError("cannot determine access range for '%s': "
                        "linear subscript(s) in '%s'"
                        % (var_name, ", ".join(
                                str(i) for i in armap.bad_subscripts)))

            n_axes_in_subscripts = set(
                    len(sub.index_tuple) for sub in armap.bad_subscripts)

            if len(n_axes_in_subscripts) != 1:
                raise RuntimeError("subscripts of '%s' with differing "
                        "numbers of axes were found" % var_name)

            n_axes, = n_axes_in_subscripts

            if n_axes == 1:
                # Leave shape undetermined--we can live with that for 1D.
                shape = None
            else:
                raise LoopyError("cannot determine access range for '%s': "
                        "undetermined index in subscript(s) '%s'"
                        % (var_name, ", ".join(
                                str(i) for i in armap.bad_subscripts)))

        else:
            # no subscripts found, let's call it a scalar
            shape = ()
    else:
        from loopy.isl_helpers import static_max_of_pw_aff
        from loopy.symbolic import pw_aff_to_expr

        shape = []
        for i in range(armap.access_range.dim(dim_type.set)):
            try:
                shape.append(
                        pw_aff_to_expr(static_max_of_pw_aff(
                            kernel.cache_manager.dim_max(
                                armap.access_range, i) + 1,
                            constants_only=False)))
            except Exception:
                print("While trying to find shape axis %d of "
                        "variable '%s', the following "
                        "exception occurred:" % (i, var_name),
                        file=sys.stderr)
                print("*** ADVICE: You may need to manually specify the "
                        "shape of argument '%s'." % (var_name),
                        file=sys.stderr)
                raise

        shape = tuple(shape)

    return shape

# }}}


# {{{ loop nest tracker

class SetTrie(object):
    """
    Similar to a trie, but uses an unordered sequence as the key.
    """

    def __init__(self, children=(), all_items=None):
        self.children = dict(children)
        # all_items should be shared within a trie.
        if all_items is None:
            self.all_items = set()
        else:
            self.all_items = all_items

    def descend(self, on_found=lambda prefix: None, prefix=frozenset()):
        on_found(prefix)
        from six import iteritems
        for prefix, child in sorted(
                iteritems(self.children),
                key=lambda it: sorted(it[0])):
            child.descend(on_found, prefix=prefix)

    def check_consistent_insert(self, items_to_insert):
        if items_to_insert & self.all_items:
            raise ValueError("inconsistent nesting")

    def add_or_update(self, key):
        if len(key) == 0:
            return

        from six import iteritems

        for child_key, child in iteritems(self.children):
            common = child_key & key
            if common:
                break
        else:
            # Key not found - insert new child
            self.check_consistent_insert(key)
            self.children[frozenset(key)] = SetTrie(all_items=self.all_items)
            self.all_items.update(key)
            return

        if child_key <= key:
            # child is a prefix of key:
            child.add_or_update(key - common)
        elif key < child_key:
            # key is a strict prefix of child:
            #
            #  -[new child]
            #     |
            #   [child]
            #
            del self.children[child_key]
            self.children[common] = SetTrie(
                children={frozenset(child_key - common): child},
                all_items=self.all_items)
        else:
            # key and child share a common prefix:
            #
            # -[new placeholder]
            #      /        \
            #  [new child]   [child]
            #
            self.check_consistent_insert(key - common)

            del self.children[child_key]
            self.children[common] = SetTrie(
                children={
                    frozenset(child_key - common): child,
                    frozenset(key - common): SetTrie(all_items=self.all_items)},
                all_items=self.all_items)
            self.all_items.update(key - common)


def get_visual_iname_order_embedding(kernel):
    """
    Return :class:`dict` `embedding` mapping inames to a totally ordered set of
    values, such that `embedding[iname1] < embedding[iname2]` when `iname2`
    is nested inside `iname1`.
    """
    from loopy.kernel.data import IlpBaseTag
    # Ignore ILP tagged inames, since they do not have to form a strict loop
    # nest.
    ilp_inames = frozenset(iname
        for iname in kernel.iname_to_tags
        if kernel.iname_tags_of_type(iname, IlpBaseTag))

    iname_trie = SetTrie()

    for insn in kernel.instructions:
        within_inames = set(
            iname for iname in insn.within_inames
            if iname not in ilp_inames)
        iname_trie.add_or_update(within_inames)

    embedding = {}

    def update_embedding(inames):
        embedding.update(
            dict((iname, (len(embedding), iname)) for iname in inames))

    iname_trie.descend(update_embedding)

    for iname in ilp_inames:
        # Nest ilp_inames innermost, so they don't interrupt visual order.
        embedding[iname] = (len(embedding), iname)

    return embedding

# }}}


# {{{ find_recursive_dependencies

def find_recursive_dependencies(kernel, insn_ids):
    queue = list(insn_ids)

    result = set(insn_ids)

    while queue:
        new_queue = []

        for insn_id in queue:
            insn = kernel.id_to_insn[insn_id]
            additionals = insn.depends_on - result
            result.update(additionals)
            new_queue.extend(additionals)

        queue = new_queue

    return result

# }}}


# {{{ find_reverse_dependencies

def find_reverse_dependencies(kernel, insn_ids):
    """Finds a set of IDs of instructions that depend on one of the insn_ids.

    :arg insn_ids: a set of instruction IDs
    """
    return frozenset(
            insn.id
            for insn in kernel.instructions
            if insn.depends_on & insn_ids)

# }}}


# {{{ draw_dependencies_as_unicode_arrows

def draw_dependencies_as_unicode_arrows(
        instructions, fore, style, flag_downward=True, max_columns=20):
    """
    :arg instructions: an ordered iterable of :class:`loopy.InstructionBase`
        instances
    :arg fore: if given, will be used like a :mod:`colorama` ``Fore`` object
        to color-code dependencies. (E.g. red for downward edges)
    :returns: A tuple ``(uniform_length, rows)``, where *rows* is a list of
        tuples (arrows, extender) with Unicode-drawn dependency arrows, one per
        entry of *instructions*. *extender* can be used to extend arrows below the
        line of an instruction. *uniform_length* is the length of the *arrows* and
        *extender* strings.
    """
    reverse_deps = {}

    for insn in instructions:
        for dep in insn.depends_on:
            reverse_deps.setdefault(dep, set()).add(insn.id)

    # mapping of to_id tuples to column_index
    dep_to_column = {}

    # {{{ find column assignments

    # mapping from column indices to (end_insn_ids, pointed_at_insn_id)
    # end_insn_ids is a set that gets modified in-place to remove 'ends'
    # (arrow origins) as they are being passed.
    columns_in_use = {}

    n_columns = [0]

    def find_free_column():
        i = 0
        while i in columns_in_use:
            i += 1
        if i+1 > n_columns[0]:
            n_columns[0] = i+1
            row.append(" ")
        return i

    def do_flag_downward(s, pointed_at_insn_id):
        if flag_downward and pointed_at_insn_id not in processed_ids:
            return fore.RED+s+style.RESET_ALL
        else:
            return s

    def make_extender():
        result = n_columns[0] * [" "]
        for col, (_, pointed_at_insn_id) in six.iteritems(columns_in_use):
            result[col] = do_flag_downward(u"│", pointed_at_insn_id)

        return result

    processed_ids = set()

    rows = []
    for insn in instructions:
        row = make_extender()

        # {{{ add rdeps for already existing columns

        rdeps = reverse_deps.get(insn.id, set()).copy() - processed_ids
        assert insn.id not in rdeps

        col = dep_to_column.get(insn.id)
        if col is not None:
            columns_in_use[col][0].update(rdeps)
        del col

        # }}}

        # {{{ add deps for already existing columns

        for dep in insn.depends_on:
            dep_key = dep
            if dep_key in dep_to_column:
                col = dep_to_column[dep]
                columns_in_use[col][0].add(insn.id)

        # }}}

        for col, (starts, pointed_at_insn_id) in list(six.iteritems(columns_in_use)):
            if insn.id == pointed_at_insn_id:
                if starts:
                    # will continue downward
                    row[col] = do_flag_downward(u">", pointed_at_insn_id)
                else:
                    # stops here

                    # placeholder, pending deletion
                    columns_in_use[col] = None

                    row[col] = do_flag_downward(u"↳", pointed_at_insn_id)

            elif insn.id in starts:
                starts.remove(insn.id)
                if starts or pointed_at_insn_id not in processed_ids:
                    # will continue downward
                    row[col] = do_flag_downward(u"├", pointed_at_insn_id)

                else:
                    # stops here
                    row[col] = u"└"
                    # placeholder, pending deletion
                    columns_in_use[col] = None

        # {{{ start arrows by reverse dep

        dep_key = insn.id
        if dep_key not in dep_to_column and rdeps:
            col = dep_to_column[dep_key] = find_free_column()
            columns_in_use[col] = (rdeps, insn.id)
            row[col] = u"↱"

        # }}}

        # {{{ start arrows by forward dep

        for dep in insn.depends_on:
            assert dep != insn.id
            dep_key = dep
            if dep_key not in dep_to_column:
                col = dep_to_column[dep_key] = find_free_column()

                # No need to add current instruction to end_insn_ids set, as
                # we're currently handling it.
                columns_in_use[col] = (set(), dep)

                row[col] = do_flag_downward(u"┌", dep)

        # }}}

        # {{{ delete columns_in_use entry for end-of-life columns

        for col, value in list(six.iteritems(columns_in_use)):
            if value is None:
                del columns_in_use[col]

        # }}}

        processed_ids.add(insn.id)

        extender = make_extender()

        rows.append(("".join(row), "".join(extender)))

    # }}}

    uniform_length = min(n_columns[0], max_columns)

    added_ellipsis = [False]

    def len_without_color_escapes(s):
        s = (s
                .replace(fore.RED, "")
                .replace(style.RESET_ALL, ""))
        return len(s)

    def truncate_without_color_escapes(s, l):
        # FIXME: This is a bit dumb--it removes color escapes when truncation
        # is needed.

        s = (s
                .replace(fore.RED, "")
                .replace(style.RESET_ALL, ""))

        return s[:l] + u"…"

    def conform_to_uniform_length(s):
        len_s = len_without_color_escapes(s)

        if len_s <= uniform_length:
            return s + " "*(uniform_length-len_s)
        else:
            added_ellipsis[0] = True
            return truncate_without_color_escapes(s, uniform_length)

    rows = [
            (conform_to_uniform_length(row),
                conform_to_uniform_length(extender))
            for row, extender in rows]

    if added_ellipsis[0]:
        uniform_length += 1

        rows = [
                (conform_to_uniform_length(row),
                    conform_to_uniform_length(extender))
                for row, extender in rows]

    return uniform_length, rows

# }}}


# {{{ stringify_instruction_list

def stringify_instruction_list(kernel):
    # {{{ topological sort

    printed_insn_ids = set()
    printed_insn_order = []

    def insert_insn_into_order(insn):
        if insn.id in printed_insn_ids:
            return
        printed_insn_ids.add(insn.id)

        for dep_id in natsorted(insn.depends_on):
            insert_insn_into_order(kernel.id_to_insn[dep_id])

        printed_insn_order.append(insn)

    for insn in kernel.instructions:
        insert_insn_into_order(insn)

    # }}}

    import loopy as lp

    Fore = kernel.options._fore  # noqa
    Style = kernel.options._style  # noqa

    uniform_arrow_length, arrows_and_extenders = \
            draw_dependencies_as_unicode_arrows(
                    printed_insn_order, fore=Fore, style=Style)

    leader = " " * uniform_arrow_length
    lines = []
    current_inames = [set()]

    if uniform_arrow_length:
        indent_level = [1]
    else:
        indent_level = [0]

    indent_increment = 2

    iname_order = kernel._get_iname_order_for_printing()

    def add_pre_line(s):
        lines.append(leader + " " * indent_level[0] + s)

    def add_main_line(s):
        lines.append(arrows + " " * indent_level[0] + s)

    def add_post_line(s):
        lines.append(extender + " " * indent_level[0] + s)

    def adapt_to_new_inames_list(new_inames):
        added = []
        removed = []

        # FIXME: Doesn't respect strict nesting
        for iname in iname_order:
            is_in_current = iname in current_inames[0]
            is_in_new = iname in new_inames

            if is_in_new == is_in_current:
                pass
            elif is_in_new and not is_in_current:
                added.append(iname)
            elif not is_in_new and is_in_current:
                removed.append(iname)
            else:
                assert False

        if removed:
            indent_level[0] -= indent_increment * len(removed)
            add_pre_line("end " + ", ".join(removed))
        if added:
            add_pre_line("for " + ", ".join(added))
            indent_level[0] += indent_increment * len(added)

        current_inames[0] = new_inames

    for insn, (arrows, extender) in zip(printed_insn_order, arrows_and_extenders):
        if isinstance(insn, lp.MultiAssignmentBase):
            lhs = ", ".join(str(a) for a in insn.assignees)
            rhs = str(insn.expression)
            trailing = []
        elif isinstance(insn, lp.CInstruction):
            lhs = ", ".join(str(a) for a in insn.assignees)
            rhs = "CODE(%s|%s)" % (
                    ", ".join(str(x) for x in insn.read_variables),
                    ", ".join("%s=%s" % (name, expr)
                        for name, expr in insn.iname_exprs))

            trailing = [l for l in insn.code.split("\n")]
        elif isinstance(insn, lp.BarrierInstruction):
            lhs = ""
            rhs = "... %sbarrier" % insn.synchronization_kind[0]
            trailing = []

        elif isinstance(insn, lp.NoOpInstruction):
            lhs = ""
            rhs = "... nop"
            trailing = []

        else:
            raise LoopyError("unexpected instruction type: %s"
                    % type(insn).__name__)

        adapt_to_new_inames_list(kernel.insn_inames(insn))

        options = ["id="+Fore.GREEN+insn.id+Style.RESET_ALL]
        if insn.priority:
            options.append("priority=%d" % insn.priority)
        if insn.tags:
            options.append("tags=%s" % ":".join(insn.tags))
        if isinstance(insn, lp.Assignment) and insn.atomicity:
            options.append("atomic=%s" % ":".join(
                str(a) for a in insn.atomicity))
        if insn.groups:
            options.append("groups=%s" % ":".join(insn.groups))
        if insn.conflicts_with_groups:
            options.append(
                    "conflicts=%s" % ":".join(insn.conflicts_with_groups))
        if insn.no_sync_with:
            options.append("no_sync_with=%s" % ":".join(
                "%s@%s" % entry for entry in sorted(insn.no_sync_with)))
        if isinstance(insn, lp.BarrierInstruction) and \
                insn.synchronization_kind == 'local':
            options.append('mem_kind=%s' % insn.mem_kind)

        if lhs:
            core = "%s = %s" % (
                Fore.CYAN+lhs+Style.RESET_ALL,
                Fore.MAGENTA+rhs+Style.RESET_ALL,
                )
        else:
            core = Fore.MAGENTA+rhs+Style.RESET_ALL

        options_str = "  {%s}" % ", ".join(options)

        if insn.predicates:
            # FIXME: precedence
            add_pre_line("if %s" % " and ".join([str(x) for x in insn.predicates]))
            indent_level[0] += indent_increment

        add_main_line(core + options_str)

        for t in trailing:
            add_post_line(t)

        if insn.predicates:
            indent_level[0] -= indent_increment
            add_post_line("end")

        leader = extender

    adapt_to_new_inames_list([])

    return lines

# }}}


# {{{ global barrier order finding

@memoize_on_first_arg
def get_global_barrier_order(kernel):
    """Return a :class:`tuple` of the listing the ids of global barrier instructions
    as they appear in order in the kernel.

    See also :class:`loopy.instruction.BarrierInstruction`.
    """
    barriers = []
    visiting = set()
    visited = set()

    unvisited = set(insn.id for insn in kernel.instructions)

    def is_barrier(my_insn_id):
        insn = kernel.id_to_insn[my_insn_id]
        from loopy.kernel.instruction import BarrierInstruction
        return isinstance(insn, BarrierInstruction) and \
            insn.synchronization_kind == "global"

    while unvisited:
        stack = [unvisited.pop()]

        while stack:
            top = stack[-1]

            if top in visiting:
                visiting.remove(top)
                if is_barrier(top):
                    barriers.append(top)

            if top in visited:
                stack.pop()
                continue

            visited.add(top)
            visiting.add(top)

            for child in kernel.id_to_insn[top].depends_on:
                # Check for no cycles.
                assert child not in visiting
                stack.append(child)

    # Ensure this is the only possible order.
    #
    # We do this by looking at the barriers in order.
    # We check for each adjacent pair (a,b) in the order if a < b,
    # i.e. if a is reachable by a chain of dependencies from b.

    visiting.clear()
    visited.clear()

    for prev_barrier, barrier in zip(barriers, barriers[1:]):
        # Check if prev_barrier is reachable from barrier.
        stack = [barrier]
        visited.discard(prev_barrier)

        while stack:
            top = stack[-1]

            if top in visiting:
                visiting.remove(top)

            if top in visited:
                stack.pop()
                continue

            visited.add(top)
            visiting.add(top)

            if top == prev_barrier:
                visiting.clear()
                break

            for child in kernel.id_to_insn[top].depends_on:
                stack.append(child)
        else:
            # Search exhausted and we did not find prev_barrier.
            raise LoopyError("barriers '%s' and '%s' are not ordered"
                             % (prev_barrier, barrier))

    return tuple(barriers)

# }}}


# {{{ find most recent global barrier

@memoize_on_first_arg
def find_most_recent_global_barrier(kernel, insn_id):
    """Return the id of the latest occuring global barrier which the
    given instruction (indirectly or directly) depends on, or *None* if this
    instruction does not depend on a global barrier.

    The return value is guaranteed to be unique because global barriers are
    totally ordered within the kernel.
    """

    global_barrier_order = get_global_barrier_order(kernel)

    if len(global_barrier_order) == 0:
        return None

    insn = kernel.id_to_insn[insn_id]

    if len(insn.depends_on) == 0:
        return None

    def is_barrier(my_insn_id):
        insn = kernel.id_to_insn[my_insn_id]
        from loopy.kernel.instruction import BarrierInstruction
        return isinstance(insn, BarrierInstruction) and \
            insn.synchronization_kind == "global"

    global_barrier_to_ordinal = dict(
            (b, i) for i, b in enumerate(global_barrier_order))

    def get_barrier_ordinal(barrier_id):
        return (global_barrier_to_ordinal[barrier_id]
                if barrier_id is not None
                else -1)

    direct_barrier_dependencies = set(
            dep for dep in insn.depends_on if is_barrier(dep))

    if len(direct_barrier_dependencies) > 0:
        return max(direct_barrier_dependencies, key=get_barrier_ordinal)
    else:
        return max((find_most_recent_global_barrier(kernel, dep)
                    for dep in insn.depends_on),
                key=get_barrier_ordinal)

# }}}


# {{{ subkernel tools

@memoize_on_first_arg
def get_subkernels(kernel):
    """Return a :class:`tuple` of the names of the subkernels in the kernel. The
    kernel must be scheduled.

    See also :class:`loopy.schedule.CallKernel`.
    """
    from loopy.kernel import KernelState
    if kernel.state != KernelState.SCHEDULED:
        raise LoopyError("Kernel must be scheduled")

    from loopy.schedule import CallKernel

    return tuple(sched_item.kernel_name
            for sched_item in kernel.schedule
            if isinstance(sched_item, CallKernel))


@memoize_on_first_arg
def get_subkernel_to_insn_id_map(kernel):
    """Return a :class:`dict` mapping subkernel names to a :class:`frozenset`
    consisting of the instruction ids scheduled within the subkernel. The
    kernel must be scheduled.
    """
    from loopy.kernel import KernelState
    if kernel.state != KernelState.SCHEDULED:
        raise LoopyError("Kernel must be scheduled")

    from loopy.schedule import (
            sched_item_to_insn_id, CallKernel, ReturnFromKernel)

    subkernel = None
    result = {}

    for sched_item in kernel.schedule:
        if isinstance(sched_item, CallKernel):
            subkernel = sched_item.kernel_name
            result[subkernel] = set()

        if isinstance(sched_item, ReturnFromKernel):
            subkernel = None

        if subkernel is not None:
            for insn_id in sched_item_to_insn_id(sched_item):
                result[subkernel].add(insn_id)

    for subkernel in result:
        result[subkernel] = frozenset(result[subkernel])

    return result

# }}}


# {{{ find aliasing equivalence classes

class DisjointSets(object):
    """
    .. automethod:: __getitem__
    .. automethod:: find_leader_or_create_group
    .. automethod:: union
    .. automethod:: union_many
    """

    # https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    def __init__(self):
        self.leader_to_group = {}
        self.element_to_leader = {}

    def __getitem__(self, item):
        """
        :arg item: A representative of an equivalence class.
        :returns: the equivalence class, given as a set of elements
        """
        try:
            leader = self.element_to_leader[item]
        except KeyError:
            return set([item])
        else:
            return self.leader_to_group[leader]

    def find_leader_or_create_group(self, el):
        try:
            return self.element_to_leader[el]
        except KeyError:
            pass

        self.element_to_leader[el] = el
        self.leader_to_group[el] = set([el])
        return el

    def union(self, a, b):
        leader_a = self.find_leader_or_create_group(a)
        leader_b = self.find_leader_or_create_group(b)

        if leader_a == leader_b:
            return

        new_leader = leader_a

        for b_el in self.leader_to_group[leader_b]:
            self.element_to_leader[b_el] = new_leader

        self.leader_to_group[leader_a].update(self.leader_to_group[leader_b])
        del self.leader_to_group[leader_b]

    def union_many(self, relation):
        """
        :arg relation: an iterable of 2-tuples enumerating the elements of the
            relation. The relation is assumed to be an equivalence relation
            (transitive, reflexive, symmetric) but need not explicitly contain
            all elements to make it that.

            The first elements of the tuples become group leaders.

        :returns: *self*
        """

        for a, b in relation:
            self.union(a, b)

        return self


def find_aliasing_equivalence_classes(kernel):
    return DisjointSets().union_many(
            (tv.base_storage, tv.name)
            for tv in six.itervalues(kernel.temporary_variables)
            if tv.base_storage is not None)

# }}}


# {{{ direction helper tools

def infer_arg_is_output_only(kernel):
    """
    Returns a copy of *kernel* with the attribute ``is_output_only`` set.

    .. note::

        If the attribute ``is_output_only`` is not supplied from an user, then
        infers it as an output argument if it is written at some point in the
        kernel.
    """
    from loopy.kernel.data import ArrayArg, ValueArg, ConstantArg, ImageArg
    new_args = []
    for arg in kernel.args:
        if isinstance(arg, (ArrayArg, ImageArg, ValueArg)):
            if arg.is_output_only is not None:
                assert isinstance(arg.is_output_only, bool)
                new_args.append(arg)
            else:
                if arg.name in kernel.get_written_variables():
                    new_args.append(arg.copy(is_output_only=True))
                else:
                    new_args.append(arg.copy(is_output_only=False))
        elif isinstance(arg, ConstantArg):
            new_args.append(arg)
        else:
            raise NotImplementedError("Unkonwn argument type %s." % type(arg))

    return kernel.copy(args=new_args)

# }}}

# vim: foldmethod=marker

"""Operations on the kernel object."""

from __future__ import division
from __future__ import absolute_import
import six

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


import numpy as np
from islpy import dim_type
from loopy.diagnostic import LoopyError

import logging
logger = logging.getLogger(__name__)


# {{{ add and infer argument dtypes

def add_dtypes(knl, dtype_dict):
    """Specify remaining unspecified argument/temporary variable types.

    :arg dtype_dict: a mapping from variable names to :class:`numpy.dtype`
        instances
    """
    dtype_dict = dtype_dict.copy()
    new_args = []

    for arg in knl.args:
        new_dtype = dtype_dict.pop(arg.name, None)
        if new_dtype is not None:
            new_dtype = np.dtype(new_dtype)
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

    if dtype_dict:
        raise RuntimeError("unused argument dtypes: %s"
                % ", ".join(dtype_dict))

    return knl.copy(args=new_args, temporary_variables=new_temp_vars)


def get_arguments_with_incomplete_dtype(knl):
    return [arg.name for arg in knl.args
            if arg.dtype is None]


def add_and_infer_dtypes(knl, dtype_dict):
    knl = add_dtypes(knl, dtype_dict)

    from loopy.preprocess import infer_unknown_types
    return infer_unknown_types(knl, expect_completion=True)

# }}}


# {{{ find_all_insn_inames fixed point iteration

def find_all_insn_inames(kernel):
    logger.debug("%s: find_all_insn_inames: start" % kernel.name)

    writer_map = kernel.writer_map()

    insn_id_to_inames = {}
    insn_assignee_inames = {}

    all_read_deps = {}
    all_write_deps = {}

    from loopy.subst import expand_subst
    kernel = expand_subst(kernel)

    for insn in kernel.instructions:
        all_read_deps[insn.id] = read_deps = insn.read_dependency_names()
        all_write_deps[insn.id] = write_deps = insn.write_dependency_names()
        deps = read_deps | write_deps

        if insn.forced_iname_deps_is_final:
            iname_deps = insn.forced_iname_deps
        else:
            iname_deps = (
                    deps & kernel.all_inames()
                    | insn.forced_iname_deps)

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

    written_vars = kernel.get_written_variables()

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

            if insn.forced_iname_deps_is_final:
                continue

            # {{{ depdency-based propagation

            # For all variables that insn depends on, find the intersection
            # of iname deps of all writers, and add those to insn's
            # dependencies.

            for tv_name in (all_read_deps[insn.id] & written_vars):
                implicit_inames = None

                for writer_id in writer_map[tv_name]:
                    writer_implicit_inames = (
                            insn_id_to_inames[writer_id]
                            - insn_assignee_inames[writer_id])
                    if implicit_inames is None:
                        implicit_inames = writer_implicit_inames
                    else:
                        implicit_inames = (implicit_inames
                                & writer_implicit_inames)

                inames_old = insn_id_to_inames[insn.id]
                inames_new = (inames_old | implicit_inames) \
                            - insn.reduction_inames()
                insn_id_to_inames[insn.id] = inames_new

                if inames_new != inames_old:
                    did_something = True
                    logger.debug("%s: find_all_insn_inames: %s -> %s (dep-based)" % (
                        kernel.name, insn.id, ", ".join(sorted(inames_new))))

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
                        inames_new.add(par)

                    # If something writes the bounds of a loop in which I'm
                    # sitting, I had better be in the inames that the writer is
                    # in.

                    if par in kernel.temporary_variables:
                        for writer_id in writer_map.get(par, []):
                            inames_new.update(insn_id_to_inames[writer_id])

            if inames_new != inames_old:
                did_something = True
                insn_id_to_inames[insn.id] = frozenset(inames_new)
                logger.debug("%s: find_all_insn_inames: %s -> %s (domain-based)" % (
                    kernel.name, insn.id, ", ".join(sorted(inames_new))))

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
        result = op(*args)
        bucket.append((set, op_name, args, result))
        return result

    def dim_min(self, set, *args):
        return self.op(set, "dim_min", set.dim_min, args)

    def dim_max(self, set, *args):
        return self.op(set, "dim_max", set.dim_max, args)

    def base_index_and_length(self, set, iname, context=None):
        if not isinstance(iname, int):
            iname_to_dim = set.space.get_var_dict()
            idx = iname_to_dim[iname][1]
        else:
            idx = iname

        lower_bound_pw_aff = self.dim_min(set, idx)
        upper_bound_pw_aff = self.dim_max(set, idx)

        from loopy.isl_helpers import static_max_of_pw_aff, static_value_of_pw_aff
        from loopy.symbolic import pw_aff_to_expr

        size = pw_aff_to_expr(static_max_of_pw_aff(
                upper_bound_pw_aff - lower_bound_pw_aff + 1, constants_only=False,
                context=context))
        try:
            base_index = pw_aff_to_expr(
                    static_value_of_pw_aff(lower_bound_pw_aff, constants_only=False,
                        context=context))
        except Exception as e:
            raise type(e)("while finding lower bound of '%s': %s" % (iname, str(e)))

        return base_index, size

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
                get_grid_sizes=self.kernel.get_grid_sizes)

# }}}


# {{{ graphviz / dot export

def get_dot_dependency_graph(kernel, iname_cluster=True, use_insn_id=False):
    """Return a string in the `dot <http://graphviz.org/>`_ language depicting
    dependencies among kernel instructions.
    """

    # make sure all automatically added stuff shows up
    from loopy.preprocess import add_default_dependencies
    kernel = add_default_dependencies(kernel)

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

    from loopy.kernel.data import ExpressionInstruction, CInstruction

    for insn in kernel.instructions:
        if isinstance(insn, ExpressionInstruction):
            op = "%s <- %s" % (insn.assignee, insn.expression)
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
        for dep in insn.insn_deps:
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
        from loopy.schedule import EnterLoop, LeaveLoop, RunInstruction, Barrier

        for sched_item in kernel.schedule:
            if isinstance(sched_item, EnterLoop):
                lines.append("subgraph cluster_%s { label=\"%s\""
                        % (sched_item.iname, sched_item.iname))
            elif isinstance(sched_item, LeaveLoop):
                lines.append("}")
            elif isinstance(sched_item, RunInstruction):
                lines.append(sched_item.insn_id)
            elif isinstance(sched_item, Barrier):
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


# {{{ domain parameter finder

class DomainParameterFinder(object):
    """Finds parameters from shapes of passed arguments."""

    def __init__(self, kernel):
        # a mapping from parameter names to a list of tuples
        # (arg_name, axis_nr, function), where function is a
        # unary function of kernel.arg_dict[arg_name].shape[axis_nr]
        # returning the desired parameter.
        self.param_to_sources = param_to_sources = {}

        param_names = kernel.all_params()

        from loopy.kernel.data import GlobalArg
        from loopy.symbolic import DependencyMapper
        from pymbolic import compile
        dep_map = DependencyMapper()

        from pymbolic import var
        for arg in kernel.args:
            if isinstance(arg, GlobalArg):
                for axis_nr, shape_i in enumerate(arg.shape):
                    deps = dep_map(shape_i)
                    if len(deps) == 1:
                        dep, = deps

                        if dep.name in param_names:
                            from pymbolic.algorithm import solve_affine_equations_for
                            try:
                                # friggin' overkill :)
                                param_expr = solve_affine_equations_for(
                                        [dep.name], [(shape_i, var("shape_i"))]
                                        )[dep.name]
                            except:
                                # went wrong? oh well
                                pass
                            else:
                                param_func = compile(param_expr, ["shape_i"])
                                param_to_sources.setdefault(dep.name, []).append(
                                        (arg.name, axis_nr, param_func))

    def __call__(self, kwargs):
        result = {}

        for param_name, sources in six.iteritems(self.param_to_sources):
            if param_name not in kwargs:
                for arg_name, axis_nr, shape_func in sources:
                    if arg_name in kwargs:
                        try:
                            shape_axis = kwargs[arg_name].shape[axis_nr]
                        except IndexError:
                            raise RuntimeError("Argument '%s' has unexpected shape. "
                                    "Tried to access axis %d (0-based), only %d "
                                    "axes present." %
                                    (arg_name, axis_nr, len(kwargs[arg_name].shape)))

                        result[param_name] = shape_func(shape_axis)
                        continue

        return result

# }}}

# vim: foldmethod=marker

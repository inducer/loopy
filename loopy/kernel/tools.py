"""Operations on the kernel object."""

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




import numpy as np
from pytools import Record, memoize_method
import islpy as isl
from islpy import dim_type

import re




# {{{ add and infer argument dtypes

def add_argument_dtypes(knl, dtype_dict):
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

    knl = knl.copy(args=new_args)

    if dtype_dict:
        raise RuntimeError("unused argument dtypes: %s"
                % ", ".join(dtype_dict))

    return knl.copy(args=new_args)

def infer_argument_dtypes(knl):
    new_args = []

    writer_map = knl.writer_map()

    from loopy.codegen.expression import (
            TypeInferenceMapper, TypeInferenceFailure)
    tim = TypeInferenceMapper(knl)

    for arg in knl.args:
        if arg.dtype is None:
            new_dtype = None

            if arg.name in knl.all_params():
                new_dtype = knl.index_dtype
            else:
                try:
                    for write_insn_id in writer_map.get(arg.name, ()):
                        write_insn = knl.id_to_insn[write_insn_id]
                        new_tim_dtype = tim(write_insn.expression)
                        if new_dtype is None:
                            new_dtype = new_tim_dtype
                        elif new_dtype != new_tim_dtype:
                            # Now we know *nothing*.
                            new_dtype = None
                            break

                except TypeInferenceFailure:
                    # Even one type inference failure is enough to
                    # make this dtype not safe to guess. Don't.
                    pass

            if new_dtype is not None:
                arg = arg.copy(dtype=new_dtype)

        new_args.append(arg)

    return knl.copy(args=new_args)

def get_arguments_with_incomplete_dtype(knl):
    return [arg.name for arg in knl.args
            if arg.dtype is None]

# }}}

# {{{ find_all_insn_inames fixed point iteration

def find_all_insn_inames(kernel):
    from loopy.symbolic import get_dependencies

    writer_map = kernel.writer_map()

    insn_id_to_inames = {}
    insn_assignee_inames = {}

    all_read_deps = {}
    all_write_deps = {}

    from loopy.subst import expand_subst
    kernel = expand_subst(kernel)

    for insn in kernel.instructions:
        all_read_deps[insn.id] = read_deps = get_dependencies(insn.expression)
        all_write_deps[insn.id] = write_deps = get_dependencies(insn.assignee)
        deps = read_deps | write_deps

        iname_deps = (
                deps & kernel.all_inames()
                | insn.forced_iname_deps)

        insn_id_to_inames[insn.id] = iname_deps
        insn_assignee_inames[insn.id] = write_deps & kernel.all_inames()

    temp_var_names = set(kernel.temporary_variables.iterkeys())

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

            # {{{ depdency-based propagation

            # For all variables that insn depends on, find the intersection
            # of iname deps of all writers, and add those to insn's
            # dependencies.

            for tv_name in (all_read_deps[insn.id] & temp_var_names):
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

            # }}}

            # {{{ domain-based propagation

            # Add all inames occurring in parameters of domains that my current
            # inames refer to.

            inames_old = insn_id_to_inames[insn.id]
            inames_new = set(insn_id_to_inames[insn.id])

            for iname in inames_old:
                home_domain = kernel.domains[kernel.get_home_domain_index(iname)]

                for par in home_domain.get_var_names(dim_type.param):
                    if par in kernel.all_inames():
                        inames_new.add(par)

            if inames_new != inames_old:
                did_something = True
                insn_id_to_inames[insn.id] = frozenset(inames_new)

            # }}}

        if not did_something:
            break

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

        for bkt_set, bkt_op, bkt_args, result  in bucket:
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
        iname_to_dim = set.space.get_var_dict()
        lower_bound_pw_aff = self.dim_min(set, iname_to_dim[iname][1])
        upper_bound_pw_aff = self.dim_max(set, iname_to_dim[iname][1])

        from loopy.isl_helpers import static_max_of_pw_aff, static_value_of_pw_aff
        from loopy.symbolic import pw_aff_to_expr

        size = pw_aff_to_expr(static_max_of_pw_aff(
                upper_bound_pw_aff - lower_bound_pw_aff + 1, constants_only=True,
                context=context))
        base_index = pw_aff_to_expr(
            static_value_of_pw_aff(lower_bound_pw_aff, constants_only=False,
                context=context))

        return base_index, size

# }}}

# {{{ domain change helper

class DomainChanger:
    """Helps change the domain responsible for *inames* within a kernel.

    .. note: Does not perform an in-place change!
    """

    def __init__(self, kernel, inames):
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

    def get_domains_with(self, replacement):
        result = self.kernel.domains[:]
        if self.leaf_domain_index is not None:
            result[self.leaf_domain_index] = replacement
        else:
            result.append(replacement)

        return result

# }}}

# {{{ graphviz / dot export

def get_dot_dependency_graph(kernel, iname_cluster=False, iname_edge=True):
    lines = []
    for insn in kernel.instructions:
        lines.append("%s [shape=\"box\"];" % insn.id)
        for dep in insn.insn_deps:
            lines.append("%s -> %s;" % (dep, insn.id))

        if iname_edge:
            for iname in kernel.insn_inames(insn):
                lines.append("%s -> %s [style=\"dotted\"];" % (iname, insn.id))

    if iname_cluster:
        for iname in kernel.all_inames():
            lines.append("subgraph cluster_%s { label=\"%s\" %s }" % (iname, iname,
                " ".join(insn.id for insn in kernel.instructions
                    if iname in kernel.insn_inames(insn))))

    return "digraph loopy_deps {\n%s\n}" % "\n".join(lines)

# }}}

# vim: foldmethod=marker

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

from loopy.symbolic import (RuleAwareIdentityMapper, SubstitutionRuleMappingContext)
from loopy.kernel.data import ValueArg, GlobalArg
import islpy as isl

__doc__ = """
.. currentmodule:: loopy

.. autofunction:: to_batched
"""


# {{{ to_batched

class _BatchVariableChanger(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, kernel, batch_varying_args,
            batch_iname_expr, sequential):
        super(_BatchVariableChanger, self).__init__(rule_mapping_context)

        self.kernel = kernel
        self.batch_varying_args = batch_varying_args
        self.batch_iname_expr = batch_iname_expr
        self.sequential = sequential

    def needs_batch_subscript(self, name):
        tv = self.kernel.temporary_variables.get(name)
        return (
                (not self.sequential
                    and (tv is not None
                        and not (
                            tv.initializer is not None
                            and tv.read_only)))
                or
                name in self.batch_varying_args)

    def map_subscript(self, expr, expn_state):
        if not self.needs_batch_subscript(expr.aggregate.name):
            return super(_BatchVariableChanger, self).map_subscript(expr, expn_state)

        idx = expr.index
        if not isinstance(idx, tuple):
            idx = (idx,)

        return type(expr)(expr.aggregate, (self.batch_iname_expr,) + idx)

    def map_variable(self, expr, expn_state):
        if not self.needs_batch_subscript(expr.name):
            return super(_BatchVariableChanger, self).map_variable(expr, expn_state)

        return expr.aggregate[self.batch_iname_expr]


def _add_unique_dim_name(name, dim_names):
    if dim_names is None:
        return dim_names

    from pytools import UniqueNameGenerator
    ng = UniqueNameGenerator(set(dim_names))
    return (ng(name),) + tuple(dim_names)


def to_batched(knl, nbatches, batch_varying_args, batch_iname_prefix="ibatch",
        sequential=False):
    """Takes in a kernel that carries out an operation and returns a kernel
    that carries out a batch of these operations.

    :arg nbatches: the number of batches. May be a constant non-negative
        integer or a string, which will be added as an integer argument.
    :arg batch_varying_args: a list of argument names that vary per-batch.
        Each such variable will have a batch index added.
    :arg sequential: A :class:`bool`. If *True*, do not duplicate
        temporary variables for each batch. This automatically tags the batch
        iname for sequential execution.
    """

    from pymbolic import var

    vng = knl.get_var_name_generator()
    batch_iname = vng(batch_iname_prefix)
    batch_iname_expr = var(batch_iname)

    new_args = []

    batch_dom_str = "{[%(iname)s]: 0 <= %(iname)s < %(nbatches)s}" % {
            "iname": batch_iname,
            "nbatches": nbatches,
            }

    if not isinstance(nbatches, int):
        batch_dom_str = "[%s] -> " % nbatches + batch_dom_str
        new_args.append(ValueArg(nbatches, dtype=knl.index_dtype))

        nbatches_expr = var(nbatches)
    else:
        nbatches_expr = nbatches

    batch_domain = isl.BasicSet(batch_dom_str)
    new_domains = [batch_domain] + knl.domains

    for arg in knl.args:
        if arg.name in batch_varying_args:
            if isinstance(arg, ValueArg):
                arg = GlobalArg(arg.name, arg.dtype, shape=(nbatches_expr,),
                        dim_tags="c")
            else:
                arg = arg.copy(
                        shape=(nbatches_expr,) + arg.shape,
                        dim_tags=("c",) * (len(arg.shape) + 1),
                        dim_names=_add_unique_dim_name("ibatch", arg.dim_names))

        new_args.append(arg)

    knl = knl.copy(
            domains=new_domains,
            args=new_args)

    if not sequential:
        new_temps = {}

        for temp in six.itervalues(knl.temporary_variables):
            if temp.initializer is not None and temp.read_only:
                new_temps[temp.name] = temp
            else:
                new_temps[temp.name] = temp.copy(
                        shape=(nbatches_expr,) + temp.shape,
                        dim_tags=("c",) * (len(temp.shape) + 1),
                        dim_names=_add_unique_dim_name("ibatch", temp.dim_names))

        knl = knl.copy(temporary_variables=new_temps)
    else:
        import loopy as lp
        from loopy.kernel.data import ForceSequentialTag
        knl = lp.tag_inames(knl, [(batch_iname, ForceSequentialTag())])

    rule_mapping_context = SubstitutionRuleMappingContext(
            knl.substitutions, vng)
    bvc = _BatchVariableChanger(rule_mapping_context,
            knl, batch_varying_args, batch_iname_expr,
            sequential=sequential)
    kernel = rule_mapping_context.finish_kernel(
            bvc.map_kernel(knl))

    batch_iname_set = frozenset([batch_iname])
    kernel = kernel.copy(
            instructions=[
                insn.copy(forced_iname_deps=insn.forced_iname_deps | batch_iname_set)
                for insn in kernel.instructions])

    return kernel

# }}}

# vim: foldmethod=marker

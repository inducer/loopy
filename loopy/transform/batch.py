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


from loopy.symbolic import (RuleAwareIdentityMapper, SubstitutionRuleMappingContext)
from loopy.kernel.data import ValueArg, ArrayArg
import islpy as isl

from loopy.translation_unit import for_each_kernel


__doc__ = """
.. currentmodule:: loopy

.. autofunction:: to_batched
"""


# {{{ to_batched

def temp_needs_batching_if_not_sequential(tv, batch_varying_args):
    from loopy.kernel.data import AddressSpace
    if tv.name in batch_varying_args:
        return True
    if tv.initializer is not None and tv.read_only:
        # do not batch read_only temps  if not in
        # `batch_varying_args`
        return False
    if tv.address_space == AddressSpace.PRIVATE:
        # do not batch private temps if not in `batch_varying args`
        return False
    return True


class _BatchVariableChanger(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, kernel, batch_varying_args,
            batch_iname_expr, sequential):
        super().__init__(rule_mapping_context)

        self.kernel = kernel
        self.batch_varying_args = batch_varying_args
        self.batch_iname_expr = batch_iname_expr
        self.sequential = sequential

    def needs_batch_subscript(self, name):
        tv = self.kernel.temporary_variables.get(name)

        if name in self.batch_varying_args:
            return True
        if not self.sequential:
            if tv is None:
                return False
            if not temp_needs_batching_if_not_sequential(tv,
                    self.batch_varying_args):
                return False

        return True

    def map_subscript(self, expr, expn_state):
        if not self.needs_batch_subscript(expr.aggregate.name):
            return super().map_subscript(expr, expn_state)

        idx = self.rec(expr.index, expn_state)
        if not isinstance(idx, tuple):
            idx = (idx,)

        return type(expr)(expr.aggregate, (self.batch_iname_expr,) + idx)

    def map_variable(self, expr, expn_state):
        if not self.needs_batch_subscript(expr.name):
            return super().map_variable(expr, expn_state)

        return expr[self.batch_iname_expr]


def _add_unique_dim_name(name, dim_names):
    if dim_names is None:
        return dim_names

    from pytools import UniqueNameGenerator
    ng = UniqueNameGenerator(set(dim_names))
    return (ng(name),) + tuple(dim_names)


@for_each_kernel
def to_batched(kernel, nbatches, batch_varying_args, batch_iname_prefix="ibatch",
        sequential=False):
    """Takes in a kernel that carries out an operation and returns a kernel
    that carries out a batch of these operations.

    .. note::
       For temporaries in a kernel that are private or read only
       globals and if `sequential=True`, loopy does not does not batch these
       variables unless explicitly mentioned in `batch_varying_args`.

    :arg nbatches: the number of batches. May be a constant non-negative
        integer or a string, which will be added as an integer argument.
    :arg batch_varying_args: a list of argument names that vary per-batch.
        Each such variable will have a batch index added.
    :arg sequential: A :class:`bool`. If *True*, do not duplicate
        temporary variables for each batch. This automatically tags the batch
        iname for sequential execution.
    """

    from pymbolic import var

    vng = kernel.get_var_name_generator()
    batch_iname = vng(batch_iname_prefix)
    batch_iname_expr = var(batch_iname)

    new_args = []

    batch_dom_str = "{{[{iname}]: 0 <= {iname} < {nbatches}}}".format(
            iname=batch_iname,
            nbatches=nbatches,
            )

    if not isinstance(nbatches, int):
        batch_dom_str = "[%s] -> " % nbatches + batch_dom_str
        new_args.append(ValueArg(nbatches, dtype=kernel.index_dtype))

        nbatches_expr = var(nbatches)
    else:
        nbatches_expr = nbatches

    batch_domain = isl.BasicSet(batch_dom_str)
    new_domains = [batch_domain] + kernel.domains

    for arg in kernel.args:
        if arg.name in batch_varying_args:
            if isinstance(arg, ValueArg):
                arg = ArrayArg(arg.name, arg.dtype, shape=(nbatches_expr,),
                        dim_tags="c")
            else:
                arg = arg.copy(
                        shape=(nbatches_expr,) + arg.shape,
                        dim_tags=("c",) * (len(arg.shape) + 1),
                        dim_names=_add_unique_dim_name("ibatch", arg.dim_names))

        new_args.append(arg)

    kernel = kernel.copy(
            domains=new_domains,
            args=new_args)

    if not sequential:
        new_temps = {}

        for temp in kernel.temporary_variables.values():
            if temp_needs_batching_if_not_sequential(temp, batch_varying_args):
                new_temps[temp.name] = temp.copy(
                        shape=(nbatches_expr,) + temp.shape,
                        dim_tags=("c",) * (len(temp.shape) + 1),
                        dim_names=_add_unique_dim_name("ibatch", temp.dim_names))
            else:
                new_temps[temp.name] = temp

        kernel = kernel.copy(temporary_variables=new_temps)
    else:
        import loopy as lp
        from loopy.kernel.data import ForceSequentialTag
        kernel = lp.tag_inames(kernel, [(batch_iname, ForceSequentialTag())])

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, vng)
    bvc = _BatchVariableChanger(rule_mapping_context,
            kernel, batch_varying_args, batch_iname_expr,
            sequential=sequential)
    kernel = rule_mapping_context.finish_kernel(
            bvc.map_kernel(kernel))

    batch_iname_set = frozenset([batch_iname])
    kernel = kernel.copy(
            instructions=[
                insn.copy(within_inames=insn.within_inames | batch_iname_set)
                for insn in kernel.instructions])

    return kernel

# }}}

# vim: foldmethod=marker

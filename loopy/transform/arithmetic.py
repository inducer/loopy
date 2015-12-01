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

from loopy.symbolic import (RuleAwareIdentityMapper,
        SubstitutionRuleMappingContext)
from loopy.diagnostic import LoopyError


# {{{ split_reduction

class _ReductionSplitter(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, within, inames, direction):
        super(_ReductionSplitter, self).__init__(
                rule_mapping_context)

        self.within = within
        self.inames = inames
        self.direction = direction

    def map_reduction(self, expr, expn_state):
        if set(expr.inames) & set(expn_state.arg_context):
            # FIXME
            raise NotImplementedError()

        if (self.inames <= set(expr.inames)
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)):
            leftover_inames = set(expr.inames) - self.inames

            from loopy.symbolic import Reduction
            if self.direction == "in":
                return Reduction(expr.operation, tuple(leftover_inames),
                        Reduction(expr.operation, tuple(self.inames),
                            self.rec(expr.expr, expn_state)))
            elif self.direction == "out":
                return Reduction(expr.operation, tuple(self.inames),
                        Reduction(expr.operation, tuple(leftover_inames),
                            self.rec(expr.expr, expn_state)))
            else:
                assert False
        else:
            return super(_ReductionSplitter, self).map_reduction(expr, expn_state)


def _split_reduction(kernel, inames, direction, within=None):
    if direction not in ["in", "out"]:
        raise ValueError("invalid value for 'direction': %s" % direction)

    if isinstance(inames, str):
        inames = inames.split(",")
    inames = set(inames)

    from loopy.context_matching import parse_stack_match
    within = parse_stack_match(within)

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    rsplit = _ReductionSplitter(rule_mapping_context,
            within, inames, direction)
    return rule_mapping_context.finish_kernel(
            rsplit.map_kernel(kernel))


def split_reduction_inward(kernel, inames, within=None):
    """Takes a reduction of the form::

        sum([i,j,k], ...)

    and splits it into two nested reductions::

        sum([j,k], sum([i], ...))

    In this case, *inames* would have been ``"i"`` indicating that
    the iname ``i`` should be made the iname governing the inner reduction.

    :arg inames: A list of inames, or a comma-separated string that can
        be parsed into those
    """

    return _split_reduction(kernel, inames, "in", within)


def split_reduction_outward(kernel, inames, within=None):
    """Takes a reduction of the form::

        sum([i,j,k], ...)

    and splits it into two nested reductions::

        sum([i], sum([j,k], ...))

    In this case, *inames* would have been ``"i"`` indicating that
    the iname ``i`` should be made the iname governing the outer reduction.

    :arg inames: A list of inames, or a comma-separated string that can
        be parsed into those
    """

    return _split_reduction(kernel, inames, "out", within)

# }}}


# {{{ fold constants

def fold_constants(kernel):
    from loopy.symbolic import ConstantFoldingMapper
    cfm = ConstantFoldingMapper()

    new_insns = [
            insn.with_transformed_expressions(cfm)
            for insn in kernel.instructions]

    new_substs = dict(
            (sub.name,
                sub.copy(expression=cfm(sub.expression)))
            for sub in six.itervalues(kernel.substitutions))

    return kernel.copy(
            instructions=new_insns,
            substitutions=new_substs)

# }}}


# {{{ collect_common_factors_on_increment

# thus far undocumented
def collect_common_factors_on_increment(kernel, var_name, is_index_specific=False):
    # FIXME: Does not understand subst rules for now
    if kernel.substitutions:
        from loopy.transform.subst import expand_subst
        kernel = expand_subst(kernel)

    from pymbolic.primitives import (Sum, Product, is_zero,
            flattened_sum, flattened_product, Subscript, Variable)
    from loopy.symbolic import get_dependencies, SubstitutionMapper

    # {{{ find common factors

    # maps lhs indices (or None for is_index_specific)
    common_factors = {}

    from loopy.kernel.data import ExpressionInstruction

    def is_assignee(insn):
        return any(
                lhs == var_name
                for lhs, sbscript in insn.assignees_and_indices())

    for insn in kernel.instructions:
        if not is_assignee(insn):
            continue

        if not isinstance(insn, ExpressionInstruction):
            raise LoopyError("'%s' modified by non-expression instruction"
                    % var_name)

        (_, index_key), = insn.assignees_and_indices()

        if not is_index_specific:
            index_key = None

        lhs = insn.assignee
        rhs = insn.expression

        if is_zero(rhs):
            continue

        if isinstance(rhs, Sum):
            sum_terms = rhs.children
        else:
            sum_terms = [rhs]

        my_common_factors = common_factors.get(index_key)

        for term in sum_terms:
            if term == lhs:
                continue

            if isinstance(term, Product):
                product_parts = set(term.children)
            else:
                product_parts = set([term])

            for part in product_parts:
                if var_name in get_dependencies(part):
                    raise LoopyError("unexpected dependency on '%s' "
                            "in RHS of instruction '%s'"
                            % (var_name, insn.id))

            if my_common_factors is None:
                my_common_factors = product_parts
            else:
                my_common_factors = my_common_factors & product_parts

        if my_common_factors is not None:
            common_factors[index_key] = my_common_factors

    # }}}

    # {{{ remove common factors

    new_insns = []

    for insn in kernel.instructions:
        if not isinstance(insn, ExpressionInstruction) or not is_assignee(insn):
            new_insns.append(insn)
            continue

        (_, index_key), = insn.assignees_and_indices()

        if not is_index_specific:
            index_key = None

        lhs = insn.assignee
        rhs = insn.expression

        if is_zero(rhs):
            new_insns.append(insn)
            continue

        if isinstance(rhs, Sum):
            sum_terms = rhs.children
        else:
            sum_terms = [rhs]

        my_common_factors = common_factors.get(index_key)
        new_sum_terms = []

        for term in sum_terms:
            if term == lhs:
                new_sum_terms.append(term)
                continue

            if isinstance(term, Product):
                product_parts = term.children
            else:
                product_parts = [term]

            new_sum_terms.append(
                    flattened_product([
                        part
                        for part in product_parts
                        if part not in my_common_factors
                        ]))

        new_insns.append(
                insn.copy(expression=flattened_sum(new_sum_terms)))

    # }}}

    # {{{ substitute common factors into usage sites

    def find_substitution(expr):
        if isinstance(expr, Subscript):
            if is_index_specific:
                index_key = expr.index
            else:
                index_key = None

            v = expr.aggregate.name
        elif isinstance(expr, Variable):
            v = expr.name
        else:
            return None

        if v != var_name:
            return None

        my_common_factors = common_factors.get(index_key)
        if my_common_factors is not None:
            return flattened_product(list(my_common_factors) + [expr])
        else:
            return expr

    insns = new_insns
    new_insns = []

    subm = SubstitutionMapper(find_substitution)

    for insn in insns:
        if not isinstance(insn, ExpressionInstruction) or is_assignee(insn):
            new_insns.append(insn)
            continue

        new_insns.append(insn.with_transformed_expressions(subm))

    # }}}

    return kernel.copy(instructions=new_insns)

# }}}


# vim: foldmethod=marker

"""
.. currentmodule:: loopy

.. autofunction:: hoist_invariant_multiplicative_terms_in_sum_reduction

.. autofunction:: extract_multiplicative_terms_in_sum_reduction_as_subst
"""

__copyright__ = "Copyright (C) 2022 Kaushik Kulkarni"

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

import pymbolic.primitives as p

from typing import (FrozenSet, TypeVar, Callable, List, Tuple, Iterable, Union, Any,
                    Optional, Sequence)
from loopy.symbolic import IdentityMapper, Reduction, CombineMapper
from loopy.kernel import LoopKernel
from loopy.kernel.data import SubstitutionRule
from loopy.diagnostic import LoopyError


# {{{ partition (copied from more-itertools)

Tpart = TypeVar("Tpart")


def partition(pred: Callable[[Tpart], bool],
              iterable: Iterable[Tpart]) -> Tuple[List[Tpart],
                                                  List[Tpart]]:
    """
    Use a predicate to partition entries into false entries and true
    entries
    """
    # Inspired from https://docs.python.org/3/library/itertools.html
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    from itertools import tee, filterfalse
    t1, t2 = tee(iterable)
    return list(filterfalse(pred, t1)), list(filter(pred, t2))

# }}}


# {{{ hoist_reduction_invariant_terms

class EinsumTermsHoister(IdentityMapper):
    """
    Mapper to hoist products out of a sum-reduction.

    .. attribute:: reduction_inames

        Inames of the reduction expressions to perform the hoisting.
    """
    def __init__(self, reduction_inames: FrozenSet[str]):
        super().__init__()
        self.reduction_inames = reduction_inames

    # type-ignore-reason: super-class.map_reduction returns 'Any'
    def map_reduction(self, expr: Reduction  # type: ignore[override]
                      ) -> p.Expression:
        if frozenset(expr.inames) != self.reduction_inames:
            return super().map_reduction(expr)

        from loopy.library.reduction import SumReductionOperation
        from loopy.symbolic import get_dependencies
        if isinstance(expr.operation, SumReductionOperation):
            if isinstance(expr.expr, p.Product):
                from pymbolic.primitives import flattened_product
                multiplicative_terms = (flattened_product(self.rec(expr.expr)
                                                          .children)
                                        .children)
            else:
                multiplicative_terms = (expr.expr,)

            invariants, variants = partition(lambda x: (get_dependencies(x)
                                                        & self.reduction_inames),
                                             multiplicative_terms)
            if not variants:
                # -> everything is invariant
                return self.rec(expr.expr) * Reduction(
                    expr.operation,
                    inames=expr.inames,
                    expr=1,  # FIXME: invalid dtype (not sure how?)
                    allow_simultaneous=expr.allow_simultaneous)
            if not invariants:
                # -> nothing to hoist
                return Reduction(
                    expr.operation,
                    inames=expr.inames,
                    expr=self.rec(expr.expr),
                    allow_simultaneous=expr.allow_simultaneous)

            return p.Product(tuple(invariants)) * Reduction(
                expr.operation,
                inames=expr.inames,
                expr=p.Product(tuple(variants)),
                allow_simultaneous=expr.allow_simultaneous)
        else:
            return super().map_reduction(expr)


def hoist_invariant_multiplicative_terms_in_sum_reduction(
    kernel: LoopKernel,
    reduction_inames: Union[str, FrozenSet[str]],
    within: Any = None
) -> LoopKernel:
    """
    Hoists loop-invariant multiplicative terms in a sum-reduction expression.

    :arg reduction_inames: The inames over which reduction is performed that defines
        the reduction expression that is to be transformed.
    :arg within: A match expression understood by :func:`loopy.match.parse_match`
        that specifies the instructions over which the transformation is to be
        performed.
    """
    from loopy.transform.instruction import map_instructions
    if isinstance(reduction_inames, str):
        reduction_inames = frozenset([reduction_inames])

    if not (reduction_inames <= kernel.all_inames()):
        raise ValueError(f"Some inames in '{reduction_inames}' not a part of"
                         " the kernel.")

    term_hoister = EinsumTermsHoister(reduction_inames)

    return map_instructions(kernel,
                            insn_match=within,
                            f=lambda x: x.with_transformed_expressions(term_hoister)
                            )

# }}}


# {{{ extract_multiplicative_terms_in_sum_reduction_as_subst

class ContainsSumReduction(CombineMapper):
    """
    Returns *True* only if the mapper maps over an expression containing a
    SumReduction operation.
    """
    def combine(self, values: Iterable[bool]) -> bool:
        return any(values)

    # type-ignore-reason: super-class.map_reduction returns 'Any'
    def map_reduction(self, expr: Reduction) -> bool:  # type: ignore[override]
        from loopy.library.reduction import SumReductionOperation
        return (isinstance(expr.operation, SumReductionOperation)
                or self.rec(expr.expr))

    def map_variable(self, expr: p.Variable) -> bool:
        return False

    def map_algebraic_leaf(self, expr: Any) -> bool:
        return False


class MultiplicativeTermReplacer(IdentityMapper):
    """
    Primary mapper of
    :func:`extract_multiplicative_terms_in_sum_reduction_as_subst`.
    """
    def __init__(self,
                 *,
                 terms_filter: Callable[[p.Expression], bool],
                 subst_name: str,
                 subst_arguments: Tuple[str, ...]) -> None:
        self.subst_name = subst_name
        self.subst_arguments = subst_arguments
        self.terms_filter = terms_filter
        super().__init__()

        # mutable state to record the expression collected by the terms_filter
        self.collected_subst_rule: Optional[SubstitutionRule] = None

    # type-ignore-reason: super-class.map_reduction returns 'Any'
    def map_reduction(self, expr: Reduction) -> Reduction:  # type: ignore[override]
        from loopy.library.reduction import SumReductionOperation
        from loopy.symbolic import SubstitutionMapper
        if isinstance(expr.operation, SumReductionOperation):
            if self.collected_subst_rule is not None:
                # => there was already a sum-reduction operation -> raise
                raise ValueError("Multiple sum reduction expressions found -> not"
                                 " allowed.")

            if isinstance(expr.expr, p.Product):
                from pymbolic.primitives import flattened_product
                terms = flattened_product(expr.expr.children).children
            else:
                terms = (expr.expr,)

            unfiltered_terms, filtered_terms = partition(self.terms_filter, terms)
            submap = SubstitutionMapper({
                argument_expr: p.Variable(f"arg{i}")
                for i, argument_expr in enumerate(self.subst_arguments)}.get)
            self.collected_subst_rule = SubstitutionRule(
                name=self.subst_name,
                arguments=tuple(f"arg{i}" for i in range(len(self.subst_arguments))),
                expression=submap(p.Product(tuple(filtered_terms))
                                  if filtered_terms
                                  else 1)
            )
            return Reduction(
                expr.operation,
                expr.inames,
                p.Product((p.Variable(self.subst_name)(*self.subst_arguments),
                           *unfiltered_terms)),
                expr.allow_simultaneous)
        else:
            return super().map_reduction(expr)


def extract_multiplicative_terms_in_sum_reduction_as_subst(
    kernel: LoopKernel,
    within: Any,
    subst_name: str,
    arguments: Sequence[p.Expression],
    terms_filter: Callable[[p.Expression], bool],
) -> LoopKernel:
    """
    Returns a copy of *kernel* with a new substitution named *subst_name* and
    *arguments* as arguments for the aggregated multiplicative terms in a
    sum-reduction expression.

    :arg within: A match expression understood by :func:`loopy.match.parse_match`
        to specify the instructions over which the transformation is to be
        performed.
    :arg terms_filter: A callable to filter which terms of the sum-reduction
        comprise the body of substitution rule.
    :arg arguments: The sub-expressions of the product of the filtered terms that
        form the arguments of the extract substitution rule in the same order.

    .. note::

        A ``LoopyError`` is raised if none or more than 1 sum-reduction expression
        appear in *within*.
    """
    from loopy.match import parse_match
    within = parse_match(within)

    matched_insns = [
        insn
        for insn in kernel.instructions
        if within(kernel, insn) and ContainsSumReduction()((insn.expression,
                                                            tuple(insn.predicates)))
    ]

    if len(matched_insns) == 0:
        raise LoopyError(f"No instructions found matching '{within}'"
                         " with sum-reductions found.")
    if len(matched_insns) > 1:
        raise LoopyError(f"More than one instruction found matching '{within}'"
                         " with sum-reductions found -> not allowed.")

    insn, = matched_insns
    replacer = MultiplicativeTermReplacer(subst_name=subst_name,
                                          subst_arguments=tuple(arguments),
                                          terms_filter=terms_filter)
    new_insn = insn.with_transformed_expressions(replacer)
    new_rule = replacer.collected_subst_rule
    new_substitutions = dict(kernel.substitutions).copy()
    if subst_name in new_substitutions:
        raise LoopyError(f"Kernel '{kernel.name}' already contains a substitution"
                         " rule named '{subst_name}'.")
    assert new_rule is not None
    new_substitutions[subst_name] = new_rule

    return kernel.copy(instructions=[new_insn if insn.id == new_insn.id else insn
                                     for insn in kernel.instructions],
                       substitutions=new_substitutions)

# }}}


# vim: foldmethod=marker

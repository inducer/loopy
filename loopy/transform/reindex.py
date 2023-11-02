"""
.. currentmodule:: loopy

.. autofunction:: reindex_temporary_using_seghir_loechner_scheme
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


import islpy as isl
from typing import Union, Iterable, Tuple
from loopy.typing import ExpressionT
from loopy.kernel import LoopKernel
from loopy.diagnostic import LoopyError
from loopy.symbolic import CombineMapper
from loopy.kernel.instruction import (MultiAssignmentBase,
                                      CInstruction, BarrierInstruction)
from loopy.symbolic import RuleAwareIdentityMapper


ISLMapT = Union[isl.BasicMap, isl.Map]
ISLSetT = Union[isl.BasicSet, isl.Set]


def _add_prime_to_dim_names(isl_map: ISLMapT,
                            dts: Iterable[isl.dim_type]) -> ISLMapT:
    """
    Returns a copy of *isl_map* with dims of types *dts* having their names
    suffixed with an apostrophe (``'``).

    .. testsetup::

        >>> import islpy as isl
        >>> from loopy.transform.reindex import _add_prime_to_dim_names

    .. doctest::

        >>> amap = isl.Map("{[i]->[j=2i]}")
        >>> _add_prime_to_dim_names(amap, [isl.dim_type.in_, isl.dim_type.out])
        Map("{ [i'] -> [j' = 2i'] }")
    """
    for dt in dts:
        for idim in range(isl_map.dim(dt)):
            old_name = isl_map.get_dim_name(dt, idim)
            new_name = f"{old_name}'"
            isl_map = isl_map.set_dim_name(dt, idim, new_name)

    return isl_map


def _get_seghir_loechner_reindexing_from_range(access_range: ISLSetT
                                               ) -> Tuple[isl.PwQPolynomial,
                                                          isl.PwQPolynomial]:
    """
    Returns ``(reindex_map, new_shape)``, where,

    * ``reindex_map`` is a quasi-polynomial of the form ``[i1, .., in] -> {f(i1,
      .., in)}`` representing that an array indexed via the subscripts
      ``[i1, ..,in]`` should be re-indexed into a 1-dimensional array as
      ``f(i1, .., in)``.
    * ``new_shape`` is a quasi-polynomial corresponding to the shape of the
      re-indexed 1-dimensional array.
    """

    # {{{ create amap: an ISL map which is an identity map from access_map's range

    amap = isl.BasicMap.identity(
        access_range
        .space
        .add_dims(isl.dim_type.in_, access_range.dim(isl.dim_type.out)))

    # set amap's dim names
    for idim in range(amap.dim(isl.dim_type.in_)):
        amap = amap.set_dim_name(isl.dim_type.in_, idim,
                                 f"_lpy_in_{idim}")
        amap = amap.set_dim_name(isl.dim_type.out, idim,
                                 f"_lpy_out_{idim}")

    amap = amap.intersect_domain(access_range)

    # }}}

    n_in = amap.dim(isl.dim_type.out)
    n_out = amap.dim(isl.dim_type.out)

    amap_lexmin = amap.lexmin()
    primed_amap_lexmin = _add_prime_to_dim_names(amap_lexmin, [isl.dim_type.in_,
                                                             isl.dim_type.out])

    lex_lt_map = isl.Map.lex_lt_map(primed_amap_lexmin, amap_lexmin)

    # make the lexmin map parametric in terms of it's previous access expressions.
    lex_lt_set = (lex_lt_map
                  .move_dims(isl.dim_type.param, 0, isl.dim_type.out, 0, n_in)
                  .domain())

    # {{{ initialize amap_to_count

    amap_to_count = _add_prime_to_dim_names(amap, [isl.dim_type.in_])
    amap_to_count = amap_to_count.insert_dims(isl.dim_type.param, 0, n_in)

    for idim in range(n_in):
        amap_to_count = amap_to_count.set_dim_name(
            isl.dim_type.param, idim,
            amap.get_dim_name(isl.dim_type.in_, idim))

    amap_to_count = amap_to_count.intersect_domain(lex_lt_set)

    # }}}

    result = amap_to_count.range().card()

    # {{{ simplify 'result' by gisting with 'access_range'

    aligned_access_range = access_range.move_dims(isl.dim_type.param, 0,
                                                  isl.dim_type.set, 0, n_out)

    for idim in range(result.dim(isl.dim_type.param)):
        aligned_access_range = (
            aligned_access_range
            .set_dim_name(isl.dim_type.param, idim,
                          result.space.get_dim_name(isl.dim_type.param,
                                                    idim)))

    result = result.gist_params(aligned_access_range.params())

    # }}}

    return result, access_range.card()


class _IndexCollector(CombineMapper):
    """
    A mapper that collects all instances of
    :class:`pymbolic.primitives.Subscript` accessing :attr:`var_name`.
    """
    def __init__(self, var_name):
        super().__init__()
        self.var_name = var_name

    def combine(self, values):
        from functools import reduce
        return reduce(frozenset.union, values, frozenset())

    def map_subscript(self, expr):
        if expr.aggregate.name == self.var_name:
            return frozenset([expr]) | super().map_subscript(expr)
        else:
            return super().map_subscript(expr)

    def map_constant(self, expr):
        return frozenset()

    map_variable = map_constant
    map_function_symbol = map_constant
    map_tagged_variable = map_constant
    map_type_cast = map_constant
    map_nan = map_constant


class ReindexingApplier(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context,
                 var_to_reindex,
                 reindexed_var_name,
                 new_index_expr,
                 index_names):

        super().__init__(rule_mapping_context)

        self.var_to_reindex = var_to_reindex
        self.reindexed_var_name = reindexed_var_name
        self.new_index_expr = new_index_expr
        self.index_names = index_names

    def map_subscript(self, expr, expn_state):
        if expr.aggregate.name != self.var_to_reindex:
            return super().map_subscript(expr, expn_state)

        from loopy.symbolic import SubstitutionMapper
        from pymbolic.mapper.substitutor import make_subst_func
        from pymbolic.primitives import Subscript, Variable

        rec_indices = tuple(self.rec(idx, expn_state) for idx in expr.index_tuple)

        assert len(self.index_names) == len(rec_indices)
        subst_func = make_subst_func(dict(zip(self.index_names, rec_indices)))

        return SubstitutionMapper(subst_func)(
            Subscript(Variable(self.reindexed_var_name),
                      self.new_index_expr)
        )


def reindex_temporary_using_seghir_loechner_scheme(kernel: LoopKernel,
                                                   var_name: str,
                                                   ) -> LoopKernel:
    """
    Returns a kernel with expressions of the form ``var_name[i1, .., in]``
    replaced with ``var_name_reindexed[f(i1, .., in)]`` where ``f`` is a
    quasi-polynomial as outlined in [Seghir_2006]_.
    """
    from loopy.transform.subst import expand_subst
    from loopy.symbolic import (BatchedAccessMapMapper, pw_qpolynomial_to_expr,
                                SubstitutionRuleMappingContext)

    if var_name not in kernel.temporary_variables:
        raise LoopyError(f"'{var_name}' not in temporary variable in kernel"
                         f" '{kernel.name}'.")

    # {{{ compute the access_range of *var_name* in *kernel*

    subst_kernel = expand_subst(kernel)
    access_map_recorder = BatchedAccessMapMapper(
        subst_kernel,
        frozenset([var_name]))

    access_exprs: Tuple[ExpressionT, ...]

    for insn in subst_kernel.instructions:
        if var_name in insn.dependency_names():
            if isinstance(insn, MultiAssignmentBase):
                access_exprs = (insn.assignees,
                                insn.expression,
                                tuple(insn.predicates))
            elif isinstance(insn, (CInstruction, BarrierInstruction)):
                access_exprs = tuple(insn.predicates)
            else:
                raise NotImplementedError(type(insn))

        access_map_recorder(access_exprs, insn.within_inames)

    vng = kernel.get_var_name_generator()
    new_var_name = vng(var_name+"_reindexed")

    access_range = access_map_recorder.get_access_range(var_name)

    del subst_kernel
    del access_map_recorder

    # }}}

    subst, new_shape = _get_seghir_loechner_reindexing_from_range(
        access_range)

    # {{{ simplify new_shape with the assumptions from kernel

    new_shape = new_shape.gist_params(kernel.assumptions)

    # }}}

    # {{{  update kernel.temporary_variables

    new_shape = new_shape.drop_unused_params()

    new_temps = dict(kernel.temporary_variables).copy()
    new_temps[new_var_name] = new_temps.pop(var_name).copy(
        name=new_var_name,
        shape=pw_qpolynomial_to_expr(new_shape),
        strides=None,
        dim_tags=None,
        dim_names=None,
    )

    kernel = kernel.copy(temporary_variables=new_temps)

    # }}}

    # {{{ perform the substitution i.e. reindex the accesses

    subst_expr = pw_qpolynomial_to_expr(subst)
    subst_dim_names = tuple(
        subst.space.get_dim_name(isl.dim_type.param, idim)
        for idim in range(access_range.dim(isl.dim_type.out)))
    assert not (set(subst_dim_names) & kernel.all_variable_names())

    rule_mapping_context = SubstitutionRuleMappingContext(kernel.substitutions,
                                                          vng)
    reindexing_mapper = ReindexingApplier(rule_mapping_context,
                                          var_name, new_var_name,
                                          subst_expr, subst_dim_names)

    def _does_access_var_name(kernel, insn, *args):
        return var_name in insn.dependency_names()

    kernel = reindexing_mapper.map_kernel(kernel,
                                          within=_does_access_var_name,
                                          map_args=False,
                                          map_tvs=False)
    kernel = rule_mapping_context.finish_kernel(kernel)

    # }}}

    # Note: Distributing a piece of code that depends on loopy and distributes
    # code that conditionally/unconditionally calls this routine does *NOT*
    # become a derivative of GPLv2. Since, as per point (0) of GPLV2 a
    # derivative is defined as: "a work containing the Program or a portion of
    # it, either verbatim or with modifications and/or translated into another
    # language."
    #
    # Loopy does *NOT* contain any portion of the barvinok library in it's
    # source code.

    return kernel

# vim: fdm=marker

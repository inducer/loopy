from __future__ import annotations


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

from collections.abc import (
    Callable,
    Collection,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from typing import TYPE_CHECKING, Any, Concatenate, Literal, TypeAlias, final
from warnings import warn

from constantdict import constantdict
from typing_extensions import deprecated, override

import islpy as isl
import pymbolic.primitives as prim
from islpy import dim_type
from pytools.tag import Tag

from loopy.diagnostic import LoopyError
from loopy.kernel import LoopKernel
from loopy.kernel.function_interface import CallableKernel
from loopy.symbolic import (
    ExpansionState,
    Reduction,
    RuleAwareIdentityMapper,
    RuleAwareSubstitutionMapper,
    SubstitutionRuleMappingContext,
    aff_from_expr,
    get_dependencies,
    pw_aff_to_expr,
)
from loopy.translation_unit import TranslationUnit, for_each_kernel
from loopy.typing import not_none


if TYPE_CHECKING:
    from pymbolic.typing import ArithmeticExpression, Expression
    from pytools import P

    from loopy.kernel.data import GroupInameTag, LocalInameTag, ToInameTagConvertible
    from loopy.kernel.instruction import InstructionBase
    from loopy.match import (
        MatchExpressionBase,
        RuleStack,
        StackMatch,
        ToMatchConvertible,
        ToStackMatchConvertible,
    )
    from loopy.typing import InameStr, InameStrSet, InsnId, ToInameStrSetConvertible


__doc__ = """
.. currentmodule:: loopy

.. autofunction:: split_iname

.. autofunction:: chunk_iname

.. autofunction:: join_inames

.. autofunction:: untag_inames

.. autofunction:: tag_inames

.. autofunction:: duplicate_inames

.. autofunction:: get_iname_duplication_options

.. autofunction:: has_schedulable_iname_nesting

.. autofunction:: prioritize_loops

.. autofunction:: rename_iname

.. autofunction:: rename_inames

.. autofunction:: remove_unused_inames

.. autofunction:: split_reduction_inward

.. autofunction:: split_reduction_outward

.. autofunction:: affine_map_inames

.. autofunction:: find_unused_axis_tag

.. autofunction:: make_reduction_inames_unique

.. autofunction:: add_inames_to_insn

.. autofunction:: map_domain

.. autofunction:: add_inames_for_unused_hw_axes

.. class:: ToInameTagConvertible

    :class:`str` or :class:`~pytools.tag.Tag`.
"""


def _to_inames_tuple(inames: ToInameStrSetConvertible) -> tuple[InameStr, ...]:
    if isinstance(inames, str):
        inames = [p for s in inames.split(",") if (p := s.strip())]

    return tuple(inames)


def _to_inames_str(inames: Collection[str]) -> str:
    return ", ".join(inames)


# {{{ set loop priority

@for_each_kernel
def prioritize_loops(
        kernel: LoopKernel,
        loop_priority: ToInameStrSetConvertible) -> LoopKernel:
    """Indicates the textual order in which loops should be entered in the
    kernel code.

    Note that this priority has an advisory role only. If the kernel logically
    requires a different nesting, the priority is ignored. Priority is only
    considered if loop nesting is ambiguous.

    :func:`prioritize_loops` can be used multiple times. If you do so, each
    given *loop_priority* specifies a scheduling constraint. The constraints
    from all calls to :func:`prioritize_loops` together establish a partial
    order on the inames (see `partially ordered sets
    <https://en.wikipedia.org/wiki/Partially_ordered_set>`__).

    :arg loop_priority: an iterable of inames, or, for brevity, a comma-separated
        string of inames.
    """
    assert isinstance(kernel, LoopKernel)

    loop_priority = _to_inames_tuple(loop_priority)
    return kernel.copy(loop_priority=kernel.loop_priority.union([loop_priority]))

# }}}


# {{{ split/chunk inames

# {{{ backend

@final
class _InameSplitter(RuleAwareIdentityMapper[[]]):
    within: MatchExpressionBase
    iname_to_split: InameStr
    outer_iname: InameStr
    inner_iname: InameStr
    replacement_index: ArithmeticExpression

    def __init__(self,
                rule_mapping_context: SubstitutionRuleMappingContext,
                within: MatchExpressionBase,
                iname_to_split: InameStr,
                outer_iname: InameStr,
                inner_iname: InameStr,
                replacement_index: ArithmeticExpression) -> None:
        super().__init__(rule_mapping_context)

        self.within = within

        self.iname_to_split = iname_to_split
        self.outer_iname = outer_iname
        self.inner_iname = inner_iname

        self.replacement_index = replacement_index

    @override
    def map_reduction(self, expr: Reduction, /,
                      expn_state: ExpansionState) -> Expression:
        if (self.iname_to_split in expr.inames
                and self.iname_to_split not in expn_state.arg_context
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction)):
            new_inames = list(expr.inames)
            new_inames.remove(self.iname_to_split)
            new_inames.extend([self.outer_iname, self.inner_iname])

            return Reduction(expr.operation, tuple(new_inames),
                        self.rec(expr.expr, expn_state),
                        expr.allow_simultaneous)
        else:
            return super().map_reduction(expr, expn_state)

    @override
    def map_variable(self, expr: prim.Variable, /,
                     expn_state: ExpansionState) -> Expression:
        if (expr.name == self.iname_to_split
                and self.iname_to_split not in expn_state.arg_context
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction)):
            return self.replacement_index
        else:
            return super().map_variable(expr, expn_state)


def _split_iname_in_set(
            s: isl.BasicSet,
            iname_to_split: InameStr,
            inner_iname: InameStr,
            outer_iname: InameStr,
            fixed_length: int,
            fixed_length_is_inner: bool
        ) -> isl.BasicSet:
    var_dict = s.get_var_dict()

    if iname_to_split not in var_dict:
        return s

    orig_dim_type, _ = var_dict[iname_to_split]
    # orig_dim_type may be set or param (the latter if the iname is
    # used as a parameter in a subdomain).

    # NB: dup_iname_to_split is not a globally valid identifier: only unique
    # wrt the set s.
    from pytools import generate_unique_names
    for dup_iname_to_split in generate_unique_names(f"dup_{iname_to_split}"):
        if dup_iname_to_split not in var_dict:
            break
    else:
        # NOTE: this should never happen, but it's a way to let pyright know
        raise ValueError(f"could not generate unique names for {iname_to_split!r}")

    from loopy.isl_helpers import duplicate_axes
    s = duplicate_axes(s, (iname_to_split,), (dup_iname_to_split,))

    outer_var_nr = s.dim(orig_dim_type)
    inner_var_nr = s.dim(orig_dim_type)+1

    s = s.add_dims(orig_dim_type, 2)
    s = s.set_dim_name(orig_dim_type, outer_var_nr, outer_iname)
    s = s.set_dim_name(orig_dim_type, inner_var_nr, inner_iname)

    from loopy.isl_helpers import make_slab

    if fixed_length_is_inner:
        fixed_iname, var_length_iname = inner_iname, outer_iname
    else:
        fixed_iname, var_length_iname = outer_iname, inner_iname

    space = s.get_space()
    s = s & (
            make_slab(space, fixed_iname, 0, fixed_length)
            # name = fixed_iname + fixed_length*var_length_iname
            .add_constraint(isl.Constraint.eq_from_names(
                space, {
                    dup_iname_to_split: 1,
                    fixed_iname: -1,
                    var_length_iname: -fixed_length})))

    dup_iname_dim_type, dup_name_idx = space.get_var_dict()[dup_iname_to_split]
    return s.project_out(dup_iname_dim_type, dup_name_idx, 1)


def _split_iname_backend(
        kernel: LoopKernel,
        iname_to_split: InameStr,
        fixed_length: int,
        fixed_length_is_inner: bool,
        make_new_loop_index: Callable[[prim.Variable, prim.Variable],
                                      ArithmeticExpression],
        outer_iname: InameStr | None = None,
        inner_iname: InameStr | None = None,
        outer_tag: ToInameTagConvertible = None,
        inner_tag: ToInameTagConvertible = None,
        slabs: tuple[int, int] = (0, 0),
        do_tagged_check: bool = True,
        within: ToMatchConvertible = None
    ) -> LoopKernel:
    """
    :arg within: if not *None*, limit the action of the transformation to
        matching contexts. See :func:`loopy.match.parse_match` for syntax.
    """

    from loopy.match import parse_match
    within = parse_match(within)

    # {{{ return the same kernel if no kernel matches

    if not any(within(kernel, insn) for insn in kernel.instructions):
        return kernel

    # }}}

    # Split inames do not inherit tags from their 'parent' inames.
    # FIXME: They *should* receive a tag that indicates that they descend from
    # an iname tagged in a certain way.
    from loopy.kernel.data import InameImplementationTag
    existing_tags = [tag
            for tag in kernel.iname_tags(iname_to_split)
            if isinstance(tag, InameImplementationTag)]

    from loopy.kernel.data import ForceSequentialTag, filter_iname_tags_by_type
    if (do_tagged_check and existing_tags
            and not filter_iname_tags_by_type(existing_tags, ForceSequentialTag)):
        raise LoopyError(f"cannot split already tagged iname '{iname_to_split}'")

    if iname_to_split not in kernel.all_inames():
        raise ValueError(
                f"cannot split loop for unknown variable '{iname_to_split}'")

    vng = kernel.get_var_name_generator()

    if outer_iname is None:
        outer_iname = vng(f"{iname_to_split}_outer")
    if inner_iname is None:
        inner_iname = vng(f"{iname_to_split}_inner")

    new_domains = [
            _split_iname_in_set(dom, iname_to_split, inner_iname, outer_iname,
                fixed_length, fixed_length_is_inner)
            for dom in kernel.domains]

    inner = prim.Variable(inner_iname)
    outer = prim.Variable(outer_iname)
    new_loop_index = make_new_loop_index(inner, outer)

    subst_map = {prim.Variable(iname_to_split): new_loop_index}

    # {{{ update within_inames

    new_insns: list[InstructionBase] = []
    for insn in kernel.instructions:
        if iname_to_split in insn.within_inames and (
                within(kernel, insn)):
            new_within_inames = (
                    (insn.within_inames - frozenset([iname_to_split]))
                    | frozenset([outer_iname, inner_iname]))
            insn = insn.copy(within_inames=new_within_inames)

        new_insns.append(insn)

    # }}}

    iname_slab_increments = kernel.iname_slab_increments.set(outer_iname, slabs)

    new_priorities: list[tuple[InameStr, ...]] = []
    for prio in kernel.loop_priority:
        new_prio = ()
        for prio_iname in prio:
            if prio_iname == iname_to_split:
                new_prio = (*new_prio, outer_iname, inner_iname)
            else:
                new_prio = (*new_prio, prio_iname)
        new_priorities.append(new_prio)

    kernel = kernel.copy(
            domains=new_domains,
            iname_slab_increments=iname_slab_increments,
            instructions=new_insns,
            applied_iname_rewrites=(*kernel.applied_iname_rewrites, subst_map),
            loop_priority=frozenset(new_priorities))

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, vng)
    ins = _InameSplitter(rule_mapping_context, within,
            iname_to_split, outer_iname, inner_iname, new_loop_index)

    from loopy.kernel.instruction import MultiAssignmentBase

    def check_insn_has_iname(kernel: LoopKernel,
                             insn: InstructionBase, *args: Any) -> bool:
        return (not isinstance(insn, MultiAssignmentBase)
                or iname_to_split in insn.dependency_names()
                or iname_to_split in insn.reduction_inames())

    kernel = ins.map_kernel(kernel, within=check_insn_has_iname,
                            map_tvs=False, map_args=False)
    kernel = rule_mapping_context.finish_kernel(kernel)

    for existing_tag in existing_tags:
        kernel = tag_inames(kernel,
                {outer_iname: existing_tag, inner_iname: existing_tag})

    kernel = tag_inames(kernel, {outer_iname: outer_tag, inner_iname: inner_tag})
    kernel = remove_unused_inames(kernel, [iname_to_split])

    return kernel

# }}}


# {{{ split iname

@for_each_kernel
def split_iname(
            kernel: LoopKernel, split_iname: InameStr, inner_length: int,
            *,
            outer_iname: InameStr | None = None,
            inner_iname: InameStr | None = None,
            outer_tag: ToInameTagConvertible = None,
            inner_tag: ToInameTagConvertible = None,
            slabs: tuple[int, int] = (0, 0),
            do_tagged_check: bool = True,
            within: ToMatchConvertible = None
        ) -> LoopKernel:
    """Split *split_iname* into two inames (an *inner* one and an *outer* one)
    so that ``split_iname == inner + outer*inner_length`` and *inner* is of
    constant length *inner_length*.

    Split inames do not inherit tags from their parent inames.

    :arg inner_length: a positive integer.
    :arg outer_iname: the new iname to use for the *inner* (fixed-length)
        loop. Defaults to a name derived from ``"{split_iname}_outer"``.
    :arg inner_iname: the new iname to use for the *inner* (fixed-length)
        loop. Defaults to a name derived from ``"{split_iname}_inner"``.
    :arg outer_tag: the iname tag (see :ref:`iname-tags`) to apply to *outer_iname*.
    :arg inner_tag: the iname tag (see :ref:`iname-tags`) to apply to *inner_iname*.
    :arg slabs: a tuple ``(head_it_count, tail_it_count)`` indicating the
        number of leading/trailing iterations of *outer_iname*
        for which separate code should be generated.
    :arg do_tagged_check: if *True*, check if the iname was already split.
    :arg within: a match, as understood by :func:`loopy.match.parse_match`.
    """
    assert isinstance(kernel, LoopKernel)

    def make_new_loop_index(
            inner: prim.Variable,
            outer: prim.Variable) -> ArithmeticExpression:
        return inner + outer*inner_length

    return _split_iname_backend(kernel, split_iname,
            fixed_length=inner_length, fixed_length_is_inner=True,
            make_new_loop_index=make_new_loop_index,
            outer_iname=outer_iname, inner_iname=inner_iname,
            outer_tag=outer_tag, inner_tag=inner_tag,
            slabs=slabs, do_tagged_check=do_tagged_check,
            within=within)

# }}}


# {{{ chunk iname

@for_each_kernel
def chunk_iname(
            kernel: LoopKernel, split_iname: InameStr, num_chunks: int,
            outer_iname: InameStr | None = None,
            inner_iname: InameStr | None = None,
            outer_tag: ToInameTagConvertible = None,
            inner_tag: ToInameTagConvertible = None,
            slabs: tuple[int, int] = (0, 0),
            do_tagged_check: bool = True,
            within: ToMatchConvertible = None
         ) -> LoopKernel:
    """Split *split_iname* into two inames (an *inner* one and an *outer* one)
    so that ``split_iname == inner + outer*chunk_length`` and *outer* is of
    fixed length *num_chunks*.

    Split inames do not inherit tags from their parent inames. See
    :func:`split_iname` for a description of the arguments.

    :arg within: a match, as understood by :func:`loopy.match.parse_match`.

    .. versionadded:: 2016.2
    """

    size = kernel.get_iname_bounds(split_iname).size
    k0 = isl.Aff.zero_on_domain(size.domain().space)
    chunk_ceil = size.div(k0+num_chunks).ceil()
    chunk_floor = size.div(k0+num_chunks).floor()
    chunk_diff = chunk_ceil - chunk_floor
    chunk_mod = size.mod_val(num_chunks)

    def make_new_loop_index(
                inner: prim.Variable,
                outer: prim.Variable) -> ArithmeticExpression:
        # These two expressions are equivalent. Benchmarking between the
        # two was inconclusive, although one is shorter.

        if 0:
            # FIXME: Triggers isl issues in check pass.
            return (
                    inner +
                    pw_aff_to_expr(chunk_floor) * outer
                    +
                    pw_aff_to_expr(chunk_diff) * prim.Min(
                        (outer, pw_aff_to_expr(chunk_mod))))
        else:
            return (
                    inner +
                    pw_aff_to_expr(chunk_ceil) * prim.Min(
                        (outer, pw_aff_to_expr(chunk_mod)))
                    +
                    pw_aff_to_expr(chunk_floor) * (
                        outer - prim.Min((outer, pw_aff_to_expr(chunk_mod)))))

    # {{{ check that iname is a box iname

    # Since the linearization used in the constraint used to map the domain
    # does not match the linearization in make_new_loop_index, we can't really
    # tolerate if the iname in question has constraints that make it non-boxy,
    # since these sub-indices would end up in the wrong spot.

    for dom in kernel.domains:
        var_dict = dom.get_var_dict()
        if split_iname not in var_dict:
            continue

        dt, idx = var_dict[split_iname]
        assert dt == dim_type.set

        aff_zero = isl.Aff.zero_on_domain(dom.space)
        aff_split_iname = aff_zero.set_coefficient_val(dim_type.in_, idx, 1)
        aligned_size = isl.align_spaces(size, aff_zero)
        box_dom = (
                dom
                .eliminate(dt, idx, 1)
                & aff_zero.le_set(aff_split_iname)
                & aff_split_iname.to_pw_aff().lt_set(aligned_size)
                )

        if not (
                box_dom <= dom.to_set() <= box_dom):
            raise LoopyError(
                    f"domain '{dom}' is not box-shape about iname "
                    f"'{split_iname}', cannot use chunk_iname()")

    # }}}

    return _split_iname_backend(kernel, split_iname,
            fixed_length=num_chunks, fixed_length_is_inner=False,
            make_new_loop_index=make_new_loop_index,
            outer_iname=outer_iname, inner_iname=inner_iname,
            outer_tag=outer_tag, inner_tag=inner_tag,
            slabs=slabs, do_tagged_check=do_tagged_check,
            within=within)

# }}}

# }}}


# {{{ join inames

@final
class _InameJoiner(RuleAwareSubstitutionMapper):
    joined_inames: frozenset[InameStr]
    new_iname: InameStr

    def __init__(self,
                rule_mapping_context: SubstitutionRuleMappingContext,
                within: StackMatch,
                subst_func: Callable[[prim.AlgebraicLeaf], Expression | None],
                joined_inames: ToInameStrSetConvertible,
                new_iname: InameStr) -> None:
        super().__init__(rule_mapping_context, subst_func, within)

        self.joined_inames = frozenset(_to_inames_tuple(joined_inames))
        self.new_iname = new_iname

    @override
    def map_reduction(self, expr: Reduction, /,
                      expn_state: ExpansionState) -> Expression:
        expr_inames = frozenset(expr.inames)
        overlap = self.joined_inames & expr_inames - set(expn_state.arg_context)
        if overlap and self.within(
                expn_state.kernel,
                expn_state.instruction,
                expn_state.stack):
            if overlap != expr_inames:
                raise LoopyError(
                        f"Cannot join inames '{_to_inames_str(self.joined_inames)}' "
                        "if there is a reduction that does not use all of the "
                        "inames being joined. (Found one with just "
                        f"'{_to_inames_str(expr_inames)}'.)")

            new_inames = (expr_inames - self.joined_inames) | {self.new_iname}
            return Reduction(expr.operation, tuple(sorted(new_inames)),
                        self.rec(expr.expr, expn_state),
                        expr.allow_simultaneous)
        else:
            return super().map_reduction(expr, expn_state)


@for_each_kernel
def join_inames(
            kernel: LoopKernel,
            inames: ToInameStrSetConvertible,
            new_iname: InameStr | None = None,
            tag: Tag | None = None,
            within: ToMatchConvertible = None) -> LoopKernel:
    """In a sense, the inverse of :func:`split_iname`.

    Takes in inames, finds their bounds (all but the first have to be bounded),
    and combines them into a single loop via analogs of ``new_iname = i0 *
    len(i1) + i1``. The old inames can be re-obtained via the appropriate
    division/modulo operations.

    :arg inames: a sequence of inames, fastest varying last.
    :arg within: a match, as understood by :func:`loopy.match.parse_match`.
    """

    if isinstance(inames, str):
        inames = [inames]

    from loopy.match import parse_match
    within = parse_match(within)

    # {{{ return the same kernel if no kernel matches

    if not any(within(kernel, insn) for insn in kernel.instructions):
        return kernel

    # }}}

    # now fastest varying first
    inames = _to_inames_tuple(inames)
    inames = inames[::-1]

    if new_iname is None:
        new_iname = kernel.get_var_name_generator()("_and_".join(inames))

    from loopy.kernel.tools import DomainChanger
    domch = DomainChanger(kernel, frozenset(inames))
    for iname in inames:
        if kernel.get_home_domain_index(iname) != domch.leaf_domain_index:
            raise LoopyError(
                    f"iname '{iname}' is not 'at home' in the join's leaf domain")

    new_domain = domch.domain
    new_dim_idx = new_domain.dim(dim_type.set)
    new_domain = new_domain.add_dims(dim_type.set, 1)
    new_domain = new_domain.set_dim_name(dim_type.set, new_dim_idx, new_iname)

    joint_aff = zero = isl.Aff.zero_on_domain(new_domain.space)
    subst_dict: dict[InameStr, ArithmeticExpression] = {}
    base_divisor = 1

    for i, iname in enumerate(inames):
        iname_dt, iname_idx = zero.get_space().get_var_dict()[iname]
        iname_aff = zero.add_coefficient_val(iname_dt, iname_idx, 1)

        joint_aff = joint_aff + base_divisor*iname_aff

        bounds = kernel.get_iname_bounds(iname, constants_only=True)

        from loopy.isl_helpers import static_max_of_pw_aff, static_value_of_pw_aff

        length = int(pw_aff_to_expr(
            static_max_of_pw_aff(bounds.size, constants_only=True)))

        try:
            lower_bound_aff = static_value_of_pw_aff(
                    bounds.lower_bound_pw_aff.coalesce(),
                    constants_only=False)
        except Exception as e:
            raise type(e)(f"while finding lower bound of '{iname}': ") from e

        my_val = prim.Variable(new_iname) // base_divisor
        if i+1 < len(inames):
            my_val %= length
        my_val += pw_aff_to_expr(lower_bound_aff)
        subst_dict[iname] = my_val

        base_divisor *= length

    from loopy.isl_helpers import iname_rel_aff
    new_domain = new_domain.add_constraint(
            isl.Constraint.equality_from_aff(
                iname_rel_aff(new_domain.get_space(), new_iname, "==", joint_aff)))

    for iname in inames:
        iname_to_dim = new_domain.get_space().get_var_dict()
        iname_dt, iname_idx = iname_to_dim[iname]

    def subst_within_inames(fid: InameStrSet) -> InameStrSet:
        result: set[InameStr] = set()
        for iname in fid:
            if iname in inames:
                result.add(new_iname)
            else:
                result.add(iname)

        return frozenset(result)

    new_insns = [
            insn.copy(within_inames=subst_within_inames(insn.within_inames))
            if within(kernel, insn) else insn
            for insn in kernel.instructions]

    kernel = kernel.copy(
            instructions=new_insns,
            domains=domch.get_domains_with(new_domain),
            applied_iname_rewrites=(*kernel.applied_iname_rewrites, subst_dict))

    from loopy.match import parse_stack_match
    within_s = parse_stack_match(within)

    from pymbolic.mapper.substitutor import make_subst_func
    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    ijoin = _InameJoiner(rule_mapping_context, within_s,
            make_subst_func(subst_dict),
            inames, new_iname)

    kernel = rule_mapping_context.finish_kernel(
            ijoin.map_kernel(kernel))

    if tag is not None:
        kernel = tag_inames(kernel, {new_iname: tag})

    return remove_unused_inames(kernel, inames)

# }}}


# {{{ untag inames

@for_each_kernel
def untag_inames(
            kernel: LoopKernel,
            iname_to_untag: InameStr,
            tag_type: type[Tag]) -> LoopKernel:
    """Remove tags on *iname_to_untag* which matches *tag_type*.

    :arg iname_to_untag: iname as string.
    :arg tag_type: a subclass of :class:`~pytools.tag.Tag`, for example a
        subclass of :class:`~loopy.kernel.data.InameImplementationTag`.

    .. versionadded:: 2018.1
    """
    from loopy.kernel.data import filter_iname_tags_by_type
    tags_to_remove = filter_iname_tags_by_type(
            kernel.inames[iname_to_untag].tags, tag_type)
    new_inames = dict(kernel.inames)
    new_inames[iname_to_untag] = kernel.inames[iname_to_untag].without_tags(
            tags_to_remove, verify_existence=False)

    return kernel.copy(inames=constantdict(new_inames))

# }}}


# {{{ tag inames

_Tags_ish: TypeAlias = Tag | Collection[Tag] | str | Collection[str] | None


@for_each_kernel
def tag_inames(
            kernel: LoopKernel,
            iname_to_tag: (
                Mapping[str, _Tags_ish]
                | Sequence[tuple[str, _Tags_ish]]
                | str),
            *,
            force: bool | None = None,
            ignore_nonexistent: bool = False
        ) -> LoopKernel:
    """Tag an iname.

    The iname to tags mapping can be given in several forms:

    * A sequence of tuples ``(iname, new_tags)``, where *new_tags* is a collection
      of instances of :class:`~pytools.tag.Tag` (for example, a subclass of
      :class:`~loopy.kernel.data.InameImplementationTag`).
    * A string of comma-separated ``iname:new_tag`` formatted entries. Note that
      in this format, it is not possible to pass in multiple tags for an iname.
    * (deprecated) a mapping of inames to collections of tags.

    *iname* may also be a wildcard using ``*`` or ``?``.

    :arg iname_to_tag: a mapping of inames to their corresponding tags in one of
        the formats described above.

    .. versionchanged:: 2016.3

        Added wildcards.

    .. versionchanged:: 2018.1

        Added iterable of tags
    """

    if force is not None:
        from warnings import warn

        warn("Setting 'force' has no effect and the argument will be removed "
             "in 2026.", DeprecationWarning, stacklevel=2)

    if isinstance(iname_to_tag, str):
        def parse_kv(s: str) -> tuple[InameStr, str]:
            colon_index = s.find(":")
            if colon_index == -1:
                raise ValueError(f"tag decl '{s}' has no colon")

            return s[:colon_index].strip(), s[colon_index+1:].strip()

        iname_to_tags_seq: Sequence[tuple[InameStr, _Tags_ish]] = [
                parse_kv(s) for s in iname_to_tag.split(",")
                if s.strip()]
    elif isinstance(iname_to_tag, Mapping):
        iname_to_tags_seq = list(iname_to_tag.items())
    else:
        iname_to_tags_seq = iname_to_tag

    if not iname_to_tag:
        return kernel

    # flatten iterables of tags for each iname

    unpack_iname_to_tag: list[tuple[InameStr, ToInameTagConvertible]] = []
    for iname, tags in iname_to_tags_seq:
        if isinstance(tags, Iterable) and not isinstance(tags, str):
            unpack_iname_to_tag.extend((iname, tag) for tag in tags)
        else:
            unpack_iname_to_tag.append((iname, tags))

    from loopy.kernel.data import parse_tag as inner_parse_tag

    def parse_tag(tag: ToInameTagConvertible) -> Iterable[Tag]:
        if isinstance(tag, str):
            if tag.startswith("like."):
                return kernel.iname_tags(tag[5:])
            elif tag == "unused.g":
                return [find_unused_axis_tag(kernel, "g")]
            elif tag == "unused.l":
                return [find_unused_axis_tag(kernel, "l")]

        result = inner_parse_tag(tag)
        if result is None:
            return []
        else:
            return [result]

    iname_to_parsed_tag = [
        (iname, subtag)
        for iname, tag in unpack_iname_to_tag
        for subtag in parse_tag(tag)
    ]

    knl_inames = dict(kernel.inames)
    all_inames = kernel.all_inames()

    from loopy.match import re_from_glob

    for iname, new_tag in iname_to_parsed_tag:
        assert new_tag is not None

        if "*" in iname or "?" in iname:
            match_re = re_from_glob(iname)
            inames = [
                sub_iname for sub_iname in all_inames
                if match_re.match(sub_iname)]
        else:
            if iname not in all_inames:
                if ignore_nonexistent:
                    continue
                else:
                    raise LoopyError(f"iname '{iname}' does not exist")

            inames = [iname]

        for sub_iname in inames:
            knl_inames[sub_iname] = knl_inames[sub_iname].tagged(new_tag)

    return kernel.copy(inames=knl_inames)

# }}}


# {{{ duplicate inames

@final
class _InameDuplicator(RuleAwareIdentityMapper[[]]):
    old_to_new: Mapping[InameStr, InameStr]
    old_inames_set: set[InameStr]
    within: StackMatch

    def __init__(self,
                 rule_mapping_context: SubstitutionRuleMappingContext,
                 old_to_new: Mapping[InameStr, InameStr],
                 within: StackMatch) -> None:
        super().__init__(rule_mapping_context)

        self.old_to_new = old_to_new
        self.old_inames_set = set(old_to_new.keys())
        self.within = within

    @override
    def map_reduction(self, expr: Reduction, /,
                      expn_state: ExpansionState) -> Expression:
        if (set(expr.inames) & self.old_inames_set
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)):
            new_inames = tuple(
                    self.old_to_new.get(iname, iname)
                    if iname not in expn_state.arg_context
                    else iname
                    for iname in expr.inames)

            return Reduction(expr.operation, new_inames,
                        self.rec(expr.expr, expn_state),
                        expr.allow_simultaneous)
        else:
            return super().map_reduction(expr, expn_state)

    @override
    def map_variable(self, expr: prim.Variable, /,
                     expn_state: ExpansionState) -> Expression:
        new_name = self.old_to_new.get(expr.name)

        if (new_name is None
                or expr.name in expn_state.arg_context
                or not self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)):
            return super().map_variable(expr, expn_state)
        else:
            return prim.Variable(new_name)

    @override
    def map_instruction(self,
                        kernel: LoopKernel,
                        insn: InstructionBase) -> InstructionBase:
        if not self.within(kernel, insn, ()):
            return insn

        new_fid = frozenset(
                self.old_to_new.get(iname, iname)
                for iname in insn.within_inames)
        return insn.copy(within_inames=new_fid)


@for_each_kernel
def duplicate_inames(kernel: LoopKernel,
                     inames: ToInameStrSetConvertible,
                     within: ToStackMatchConvertible,
                     new_inames: InameStr | Sequence[InameStr | None] | None = None,
                     suffix: str | None = None,
                     tags: Mapping[str, Tag] | None = None) -> LoopKernel:
    """
    :arg within: a stack match as understood by :func:`loopy.match.parse_stack_match`.
    """
    if tags is None:
        tags = {}

    # {{{ normalize arguments, find unique new_inames

    inames = _to_inames_tuple(inames)

    if new_inames is None:
        new_inames = [None] * len(inames)
    else:
        new_inames = _to_inames_tuple(new_inames)

    if len(new_inames) != len(inames):
        raise ValueError(
            "'new_inames' must have the same number of entries as 'inames': "
            f"got {len(new_inames)} (expected {len(inames)})")

    from loopy.match import parse_stack_match
    stack = parse_stack_match(within)

    name_gen = kernel.get_var_name_generator()
    new_suffixed_inames: list[InameStr] = []
    for i, iname in enumerate(inames):
        new_iname = new_inames[i]

        if new_iname is None:
            new_iname = iname

            if suffix is not None:
                new_iname += suffix

            new_iname = name_gen(new_iname)
        else:
            if name_gen.is_name_conflicting(new_iname):
                raise ValueError(
                    f"new iname '{new_iname}' conflicts with existing names")

            name_gen.add_name(new_iname)

        new_suffixed_inames.append(new_iname)

    # }}}

    # {{{ duplicate the inames

    from loopy.isl_helpers import duplicate_axes
    from loopy.kernel.tools import DomainChanger

    for old_iname, new_iname in zip(inames, new_suffixed_inames, strict=True):
        domch = DomainChanger(kernel, frozenset([old_iname]))

        dup_iname = duplicate_axes(domch.domain, [old_iname], [new_iname])
        kernel = kernel.copy(domains=domch.get_domains_with(dup_iname))

    # }}}

    # {{{ change the inames in the code

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, name_gen)
    indup = _InameDuplicator(rule_mapping_context,
            old_to_new=dict(list(zip(inames, new_suffixed_inames, strict=True))),
            within=stack)

    def _does_access_old_inames(kernel: LoopKernel,
                                insn: InstructionBase,
                                *args: Any) -> bool:
        all_inames = (insn.within_inames
                      | insn.reduction_inames()
                      | insn.sub_array_ref_inames())
        return bool(frozenset(inames) & all_inames)

    kernel = rule_mapping_context.finish_kernel(
            indup.map_kernel(kernel, within=_does_access_old_inames,
                             map_tvs=False, map_args=False))

    # }}}

    # {{{ realize tags

    for old_iname, new_iname in zip(inames, new_suffixed_inames, strict=True):
        new_tag = tags.get(old_iname)
        if new_tag is not None:
            kernel = tag_inames(kernel, {not_none(new_iname): new_tag})

    # }}}

    return kernel

# }}}


# {{{ iname duplication for schedulability


def _get_iname_duplication_options(
        insn_iname_sets: frozenset[InameStrSet],
        old_common_inames: InameStrSet | None = None,
    ) -> Iterator[tuple[InameStr, tuple[InameStrSet, ...]]]:
    if old_common_inames is None:
        old_common_inames = frozenset()

    # Remove common inames of the current insn_iname_sets, as they are not relevant
    # for splitting.
    common = frozenset[str]().union(*insn_iname_sets).intersection(*insn_iname_sets)

    # If common inames were found, we reduce the problem and go into recursion
    if common:
        # Remove the common inames from the instruction dependencies
        insn_iname_sets = (
            frozenset(iname_set - common for iname_set in insn_iname_sets)
            -
            frozenset([frozenset[str]()]))
        # Join the common inames with those previously found
        common = common.union(old_common_inames)

        # Go into recursion
        yield from _get_iname_duplication_options(insn_iname_sets, common)
        # Do not yield anything beyond here!
        return

    # Try finding a partitioning of the remaining inames, such that all instructions
    # use only inames from one of the disjoint sets from the partitioning.
    def join_sets_if_not_disjoint(
            sets: frozenset[InameStrSet]
        ) -> tuple[frozenset[InameStrSet], bool]:
        for s1 in sets:
            for s2 in sets:
                if s1 != s2 and s1 & s2:
                    return (
                        (sets - frozenset([s1, s2]))
                        | frozenset([s1 | s2])
                        ), False

        return sets, True

    partitioning = insn_iname_sets
    stop = False
    while not stop:
        partitioning, stop = join_sets_if_not_disjoint(partitioning)

    # If a partitioning was found we recursively apply this algorithm to the
    # subproblems
    if len(partitioning) > 1:
        for part in partitioning:
            working_set = frozenset(s for s in insn_iname_sets if s <= part)
            yield from _get_iname_duplication_options(working_set, old_common_inames)

    # If exactly one set was found, an iname duplication is necessary
    elif len(partitioning) == 1:
        inames, = partitioning

        # There are splitting options for all inames
        for iname in inames:
            iname_insns = frozenset(
                    insn
                    for insn in insn_iname_sets
                    if frozenset([iname]) <= insn)

            import itertools as it
            # For a given iname, the set of instructions containing this iname
            # is inspected.  For each element of the power set without the
            # empty and the full set, one duplication option is generated.
            for insns_to_dup in it.chain.from_iterable(
                    it.combinations(iname_insns, i)
                    for i in range(1, len(iname_insns))):
                yield (
                    iname,
                    tuple(insn | old_common_inames for insn in insns_to_dup))

    # If partitioning was empty, we have recursed successfully and yield nothing


def get_iname_duplication_options(
            kernel: LoopKernel
    ) -> Iterator[tuple[InameStr, MatchExpressionBase]]:
    """List options for duplication of inames, if necessary for schedulability.

    :returns: a generator listing all options to duplicate inames, if duplication
        of an iname is necessary to ensure the schedulability of the kernel.
        Duplication options are returned as tuples (iname, within) as
        understood by :func:`duplicate_inames`. There is no guarantee that the
        transformed kernel will be schedulable, because multiple duplications
        of iname may be necessary.

    Some kernels require the duplication of inames in order to be schedulable, as the
    forced iname dependencies define an over-determined problem to the scheduler.
    Consider the following minimal example::

        knl = lp.make_kernel(["{[i,j]:0<=i,j<n}"],
                             \"\"\"
                             mat1[i,j] = mat1[i,j] + 1 {inames=i:j, id=i1}
                             mat2[j] = mat2[j] + 1 {inames=j, id=i2}
                             mat3[i] = mat3[i] + 1 {inames=i, id=i3}
                             \"\"\")

    In the example, there are four possibilities to resolve the problem:
    * duplicating i in instruction i3
    * duplicating i in instruction i1 and i3
    * duplicating j in instruction i2
    * duplicating i in instruction i2 and i3

    Use :func:`has_schedulable_iname_nesting` to decide whether an iname needs to be
    duplicated in a given kernel.
    """
    if isinstance(kernel, TranslationUnit):
        if len([clbl for clbl in kernel.callables_table.values()
                if isinstance(clbl, CallableKernel)]) == 1:
            kernel = kernel[next(iter(kernel.entrypoints))]

    assert isinstance(kernel, LoopKernel)

    from loopy.kernel.data import ConcurrentTag

    concurrent_inames = {
            iname
            for iname in kernel.all_inames()
            if kernel.iname_tags_of_type(iname, ConcurrentTag)}

    # First we extract the minimal necessary information from the kernel
    insn_iname_sets = (
        frozenset(
            insn.within_inames - concurrent_inames
            for insn in kernel.instructions)
        -
        frozenset([frozenset[str]()]))

    # Get the duplication options as a tuple of iname and a set
    for iname, insns in _get_iname_duplication_options(insn_iname_sets):
        # Check whether this iname has a parallel tag and discard it if so
        if kernel.iname_tags_of_type(iname, ConcurrentTag):
            continue

        # Reconstruct an object that may be passed to the within parameter of
        # loopy.duplicate_inames
        from loopy.match import Id, Or
        within = Or(tuple(
            Id(insn.id) for insn in kernel.instructions
            if insn.within_inames in insns))

        # Only yield the result if an instruction matched.
        if within.children:
            yield iname, within


def has_schedulable_iname_nesting(kernel: LoopKernel) -> bool:
    """
    :returns: a :class:`bool` indicating whether this kernel needs
        an iname duplication in order to be schedulable.
    """
    if isinstance(kernel, TranslationUnit):
        if len([clbl for clbl in kernel.callables_table.values()
                if isinstance(clbl, CallableKernel)]) == 1:
            kernel = kernel[next(iter(kernel.entrypoints))]

    return not bool(next(get_iname_duplication_options(kernel), False))

# }}}


# {{{ remove unused inames

def get_used_inames(kernel: LoopKernel) -> InameStrSet:
    import loopy as lp

    exp_kernel = lp.expand_subst(kernel)

    used_inames: set[InameStr] = set()
    for insn in exp_kernel.instructions:
        used_inames.update(
                insn.within_inames
                | insn.reduction_inames()
                | insn.sub_array_ref_inames())

    return frozenset(used_inames)


@for_each_kernel
def remove_unused_inames(
        kernel: LoopKernel,
        inames: ToInameStrSetConvertible | None = None
    ) -> LoopKernel:
    """Delete those among *inames* that are unused, i.e. project them out of the domain.

    If these inames pose implicit restrictions on other inames, these
    restrictions will persist as existentially quantified variables.

    :arg inames: may be an iterable of inames or a string of comma-separated inames.
    """

    # {{{ normalize arguments

    inames = (
            kernel.all_inames()
            if inames is None
            else frozenset(_to_inames_tuple(inames)))

    # }}}

    # {{{ check which inames are unused

    unused_inames = inames - get_used_inames(kernel)

    # }}}

    # {{{ remove them

    domains = kernel.domains
    for iname in sorted(unused_inames):
        new_domains: list[isl.BasicSet] = []

        for dom in domains:
            try:
                dt, idx = dom.get_var_dict()[iname]
            except KeyError:
                pass
            else:
                dom = dom.project_out(dt, idx, 1)
            new_domains.append(dom)

        domains = new_domains

    return kernel.copy(domains=domains)

    # }}}


def remove_any_newly_unused_inames(
        transformation_func: Callable[Concatenate[LoopKernel, P], LoopKernel]
    ) -> Callable[Concatenate[LoopKernel, P], LoopKernel]:
    from functools import wraps

    @wraps(transformation_func)
    def wrapper(kernel: LoopKernel, *args: P.args, **kwargs: P.kwargs) -> LoopKernel:
        # check for remove_unused_inames argument, default: True
        remove_newly_unused_inames = kwargs.pop("remove_newly_unused_inames", True)

        if remove_newly_unused_inames:
            # call transform
            transformed_kernel = transformation_func(kernel, *args, **kwargs)

            if transformed_kernel is kernel:
                return kernel

            # determine which inames were already unused
            inames_already_unused = kernel.all_inames() - get_used_inames(kernel)

            # Remove inames that are unused due to transform
            return remove_unused_inames(
                transformed_kernel,
                transformed_kernel.all_inames()-inames_already_unused)
        else:
            # call transform
            return transformation_func(kernel, *args, **kwargs)

    return wrapper

# }}}


# {{{ split_reduction

@final
class _ReductionSplitter(RuleAwareIdentityMapper[[]]):
    within: StackMatch
    inames: InameStrSet
    direction: Literal["in", "out"]

    def __init__(self,
                 rule_mapping_context: SubstitutionRuleMappingContext,
                 within: StackMatch,
                 inames: InameStrSet,
                 direction: Literal["in", "out"]) -> None:
        super().__init__(rule_mapping_context)

        self.within = within
        self.inames = inames
        self.direction = direction

    @override
    def map_reduction(self, expr: Reduction, /,
                      expn_state: ExpansionState) -> Expression:
        if set(expr.inames) & set(expn_state.arg_context):
            # FIXME
            raise NotImplementedError()

        if (self.inames <= set(expr.inames)
                and self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)):
            leftover_inames = set(expr.inames) - self.inames

            if self.direction == "in":
                return Reduction(expr.operation, tuple(leftover_inames),
                        Reduction(expr.operation, tuple(self.inames),
                            self.rec(expr.expr, expn_state),
                            expr.allow_simultaneous),
                        expr.allow_simultaneous)
            elif self.direction == "out":
                return Reduction(expr.operation, tuple(self.inames),
                        Reduction(expr.operation, tuple(leftover_inames),
                            self.rec(expr.expr, expn_state),
                            expr.allow_simultaneous),
                        expr.allow_simultaneous)
            else:
                raise ValueError(f"unsupported direction: {self.direction!r}")
        else:
            return super().map_reduction(expr, expn_state)


def _split_reduction(
            kernel: LoopKernel,
            inames: ToInameStrSetConvertible,
            direction: Literal["in", "out"],
            within: ToStackMatchConvertible = None
        ) -> LoopKernel:
    if direction not in ["in", "out"]:
        raise ValueError(f"invalid value for 'direction': {direction!r}")

    inames = frozenset(_to_inames_tuple(inames))
    if not (inames <= kernel.all_inames()):
        raise LoopyError(f"unknown inames: {inames - kernel.all_inames()}.")

    from loopy.match import parse_stack_match
    within = parse_stack_match(within)

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    rsplit = _ReductionSplitter(rule_mapping_context,
            within, inames, direction)
    return rule_mapping_context.finish_kernel(
            rsplit.map_kernel(kernel))


@for_each_kernel
def split_reduction_inward(
            kernel: LoopKernel,
            inames: ToInameStrSetConvertible,
            within: ToStackMatchConvertible = None) -> LoopKernel:
    """Takes a reduction of the form::

        sum([i, j, k], ...)

    and splits it into two nested reductions::

        sum([j, k], sum([i], ...))

    In this case, *inames* would have been ``"i"`` indicating that
    the iname ``i`` should be made the iname governing the inner reduction.

    :arg inames: an iterable of inames, or a comma-separated string that can
        be parsed into those.
    """

    return _split_reduction(kernel, inames, "in", within)


@for_each_kernel
def split_reduction_outward(
            kernel: LoopKernel,
            inames: ToInameStrSetConvertible,
            within: ToStackMatchConvertible = None) -> LoopKernel:
    """Takes a reduction of the form::

        sum([i, j, k], ...)

    and splits it into two nested reductions::

        sum([i], sum([j, k], ...))

    In this case, *inames* would have been ``"i"`` indicating that
    the iname ``i`` should be made the iname governing the outer reduction.

    :arg inames: an iterable of inames, or a comma-separated string that can
        be parsed into those.
    """

    return _split_reduction(kernel, inames, "out", within)

# }}}


# {{{ affine map inames

@deprecated("use map_domain instead")
@for_each_kernel
def affine_map_inames(
            kernel: LoopKernel,
            old_inames: ToInameStrSetConvertible,
            new_inames: ToInameStrSetConvertible,
            equations:
                Sequence[tuple[ArithmeticExpression, ArithmeticExpression] | str]
        ) -> LoopKernel:
    """Return a new *kernel* where the affine transform specified by *equations*
    has been applied to the inames.

    :arg old_inames: an iterable of inames to be replaced by affine transforms
        of their values. May also be a string of comma-separated inames.
    :arg new_inames: an iterable of new inames that are not yet used in *kernel*,
        but have their values established in terms of *old_inames* by
        *equations*. May also be a string of comma-separated inames.
    :arg equations: a sequence of equations establishing a relationship
        between *old_inames* and *new_inames*. Each equation should be
        a tuple ``(lhs, rhs)`` of expressions or a string, with left and
        right hand side of the equation separated by ``=``.
    """

    warn("affine_map_inames is deprecated and will stop working in 2H2026. "
         "Rewrite in terms of map_domain.", DeprecationWarning, stacklevel=3)

    # {{{ check and parse arguments

    old_inames = _to_inames_tuple(old_inames)
    new_inames = _to_inames_tuple(new_inames)
    if isinstance(equations, str):
        equations = [equations]

    import re
    eqn_re = re.compile(r"^([^=]+)=([^=]+)$")

    def parse_equation(
            eqn: str | tuple[ArithmeticExpression, ArithmeticExpression]
        ) -> tuple[ArithmeticExpression, ArithmeticExpression]:
        if isinstance(eqn, str):
            eqn_match = eqn_re.match(eqn)
            if not eqn_match:
                raise ValueError(f"invalid equation: {eqn}")

            from loopy.symbolic import parse
            lhs = parse(eqn_match.group(1))
            rhs = parse(eqn_match.group(2))
            assert prim.is_arithmetic_expression(lhs)
            assert prim.is_arithmetic_expression(rhs)
            return (lhs, rhs)
        elif isinstance(eqn, tuple):
            if len(eqn) != 2:
                raise ValueError(
                    "unexpected length of equation tuple, got {len(eqn)}, should be 2")

            return eqn
        else:
            raise TypeError(
                f"unexpected type of equation: got {type(eqn)}, "
                "should be string or tuple")

    equations = [parse_equation(eqn) for eqn in equations]

    all_vars = kernel.all_variable_names()
    for iname in new_inames:
        if iname in all_vars:
            raise LoopyError(f"new iname '{iname}' is already used in kernel")

    for iname in old_inames:
        if iname not in kernel.all_inames():
            raise LoopyError(f"old iname '{iname}' not known")

    # }}}

    # {{{ substitute iname use

    from pymbolic.algorithm import solve_affine_equations_for
    old_inames_to_expr = solve_affine_equations_for(old_inames, equations)

    subst_dict: dict[str, ArithmeticExpression] = {
            v.name: expr
            for v, expr in old_inames_to_expr.items()}

    var_name_gen = kernel.get_var_name_generator()

    from pymbolic.mapper.substitutor import make_subst_func

    from loopy.match import parse_stack_match

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, var_name_gen)
    old_to_new = RuleAwareSubstitutionMapper(rule_mapping_context,
            make_subst_func(subst_dict), within=parse_stack_match(None))

    kernel = rule_mapping_context.finish_kernel(old_to_new.map_kernel(kernel))
    kernel = kernel.copy(
            applied_iname_rewrites=(*kernel.applied_iname_rewrites, subst_dict))

    # }}}

    # {{{ change domains

    new_inames_set = frozenset(new_inames)
    old_inames_set = frozenset(old_inames)

    new_domains: list[isl.BasicSet] = []
    for idom, dom in enumerate(kernel.domains):
        dom_var_dict = dom.get_var_dict()
        old_iname_overlap = [
                iname
                for iname in old_inames
                if iname in dom_var_dict]

        if not old_iname_overlap:
            new_domains.append(dom)
            continue

        dom_new_inames: set[InameStr] = set()
        dom_old_inames: set[InameStr] = set()

        # mapping for new inames to dim_types
        new_iname_dim_types: dict[str, isl.dim_type] = {}

        dom_equations: list[tuple[ArithmeticExpression, ArithmeticExpression]] = []
        for iname in old_iname_overlap:
            for ieqn, (lhs, rhs) in enumerate(equations):
                eqn_deps = get_dependencies(lhs) | get_dependencies(rhs)
                if iname in eqn_deps:
                    dom_new_inames.update(eqn_deps & new_inames_set)
                    dom_old_inames.update(eqn_deps & old_inames_set)

                if dom_old_inames:
                    dom_equations.append((lhs, rhs))

                this_eqn_old_iname_dim_types = {
                        dom_var_dict[old_iname][0]
                        for old_iname in eqn_deps & old_inames_set}

                if this_eqn_old_iname_dim_types:
                    if len(this_eqn_old_iname_dim_types) > 1:
                        raise ValueError(
                                f"inames '{_to_inames_str(eqn_deps & old_inames_set)}' "
                                f"(from equation {ieqn} (0-based)) in domain {idom} "
                                "(0-based) are not of a uniform dim_type")

                    this_eqn_new_iname_dim_type, = this_eqn_old_iname_dim_types

                    for new_iname in eqn_deps & new_inames_set:
                        if new_iname in new_iname_dim_types:
                            if (this_eqn_new_iname_dim_type
                                    != new_iname_dim_types[new_iname]):
                                raise ValueError(
                                        "dim_type disagreement for iname "
                                        f"'{new_iname}' (from equation {ieqn} "
                                        f"(0-based)) in domain {idom} (0-based)")
                        else:
                            new_iname_dim_types[new_iname] = \
                                    this_eqn_new_iname_dim_type

        if not dom_old_inames <= set(dom_var_dict):
            raise ValueError(
                    f"domain {idom} (0-based) does not know about all old inames "
                    f"(specifically {_to_inames_str(dom_old_inames-set(dom_var_dict))}"
                    ") needed to define new inames")

        # add inames to domain with correct dim_types
        for iname in dom_new_inames:
            dt = new_iname_dim_types[iname]
            iname_idx = dom.dim(dt)
            dom = dom.add_dims(dt, 1)
            dom = dom.set_dim_name(dt, iname_idx, iname)

        # add equations
        for lhs, rhs in dom_equations:
            dom = dom.add_constraint(
                    isl.Constraint.equality_from_aff(
                        aff_from_expr(dom.space, rhs - lhs)))

        # project out old inames
        for iname in dom_old_inames:
            dt, idx = dom.get_var_dict()[iname]
            dom = dom.project_out(dt, idx, 1)

        new_domains.append(dom)

    # }}}

    # {{{ switch iname refs in instructions

    def fix_iname_set(insn_id: InsnId, inames: InameStrSet) -> InameStrSet:
        if old_inames_set <= inames:
            return (inames - old_inames_set) | new_inames_set
        elif old_inames_set & inames:
            raise LoopyError(
                    f"instruction '{insn_id}' uses only a part "
                    f"({_to_inames_str(old_inames_set & inames)}), not all, "
                    "of the old inames")
        else:
            return inames

    new_instructions = [
            insn.copy(within_inames=fix_iname_set(insn.id, insn.within_inames))
            for insn in kernel.instructions]

    # }}}

    return kernel.copy(domains=new_domains, instructions=new_instructions)

# }}}


# {{{ find unused axes

def find_unused_axis_tag(
                kernel: LoopKernel,
                kind: Literal["l", "g"] | type[GroupInameTag | LocalInameTag],
                insn_match: ToMatchConvertible = None,
            ) -> GroupInameTag | LocalInameTag:
    """For one of the hardware-parallel execution tags, find an unused axis.

    :arg insn_match: a match as understood by :func:`loopy.match.parse_match`.
    :arg kind: may be "l" or "g", or the corresponding tag class name.

    :returns: an :class:`~loopy.kernel.data.GroupInameTag` or
        :class:`~loopy.kernel.data.LocalInameTag` that is not being used within
        the instructions matched by *insn_match*.
    """
    from loopy.kernel.data import GroupInameTag, LocalInameTag

    kind_cls: type[GroupInameTag | LocalInameTag]
    if isinstance(kind, str):
        for cls in [GroupInameTag, LocalInameTag]:
            if kind == cls.print_name:
                kind_cls = cls
                break
        else:
            raise LoopyError(f"invalid tag kind: {kind!r}")
    else:
        kind_cls = kind

    from loopy.match import parse_match
    match = parse_match(insn_match)
    insns = [insn for insn in kernel.instructions if match(kernel, insn)]

    used_axes: set[int] = set()
    for insn in insns:
        for iname in insn.within_inames:
            if kernel.iname_tags_of_type(iname, kind_cls):
                used_axes.add(kind_cls.axis)

    i = 0
    while i in used_axes:
        i += 1

    return kind_cls(i)

# }}}


# {{{ separate_loop_head_tail_slab

# undocumented, because not super-useful
def separate_loop_head_tail_slab(kernel: LoopKernel,
                                 iname: InameStr,
                                 head_it_count: int,
                                 tail_it_count: int) -> LoopKernel:
    """Mark *iname* so that the separate code is generated for
    the lower *head_it_count* and the upper *tail_it_count*
    iterations of the loop on *iname*.
    """

    iname_slab_increments = dict(kernel.iname_slab_increments)
    iname_slab_increments[iname] = (head_it_count, tail_it_count)

    return kernel.copy(iname_slab_increments=constantdict(iname_slab_increments))

# }}}


# {{{ make_reduction_inames_unique

@final
class _ReductionInameUniquifier(RuleAwareIdentityMapper[[]]):
    inames: InameStrSet | None
    old_to_new: list[tuple[str, str]]
    within: StackMatch

    iname_to_red_count: dict[InameStr, int]
    iname_to_nonsimultaneous_red_count: dict[InameStr, int]

    def __init__(self,
                 rule_mapping_context: SubstitutionRuleMappingContext,
                 inames: InameStrSet | None,
                 within: StackMatch) -> None:
        super().__init__(rule_mapping_context)

        self.inames = inames
        self.old_to_new = []
        self.within = within

        self.iname_to_red_count = {}
        self.iname_to_nonsimultaneous_red_count = {}

    @override
    def get_cache_key(self, expr: Expression, expn_state: ExpansionState) -> Hashable:
        return (super().get_cache_key(expr, expn_state),
                hash(frozenset(self.iname_to_red_count.items())),
                hash(frozenset(self.iname_to_nonsimultaneous_red_count.items())),
                )

    @override
    def map_reduction(self, expr: Reduction, /,
                      expn_state: ExpansionState) -> Expression:
        within = self.within(
                    expn_state.kernel,
                    expn_state.instruction,
                    expn_state.stack)

        for iname in expr.inames:
            self.iname_to_red_count[iname] = (
                    self.iname_to_red_count.get(iname, 0) + 1)
            if not expr.allow_simultaneous:
                self.iname_to_nonsimultaneous_red_count[iname] = (
                    self.iname_to_nonsimultaneous_red_count.get(iname, 0) + 1)

        if within and not expr.allow_simultaneous:
            subst_dict: dict[InameStr, prim.Variable] = {}

            new_inames: list[str] = []
            for iname in expr.inames:
                if (
                        not (self.inames is None or iname in self.inames)
                        or
                        self.iname_to_red_count[iname] <= 1):
                    new_inames.append(iname)
                    continue

                new_iname = self.rule_mapping_context.make_unique_var_name(iname)
                subst_dict[iname] = prim.Variable(new_iname)
                self.old_to_new.append((iname, new_iname))
                new_inames.append(new_iname)

            from pymbolic.mapper.substitutor import make_subst_func

            from loopy.symbolic import SubstitutionMapper
            return Reduction(expr.operation, tuple(new_inames),
                    self.rec(
                        SubstitutionMapper(make_subst_func(subst_dict))(expr.expr),
                        expn_state),
                    expr.allow_simultaneous)
        else:
            return super().map_reduction(expr, expn_state)


@for_each_kernel
def make_reduction_inames_unique(
        kernel: LoopKernel,
        inames: ToInameStrSetConvertible | None = None,
        within: ToStackMatchConvertible | None = None) -> LoopKernel:
    """
    :arg inames: if not *None*, only apply to these inames.
    :arg within: a stack match as understood by :func:`loopy.match.parse_stack_match`.

    .. versionadded:: 2016.2
    """

    name_gen = kernel.get_var_name_generator()

    from loopy.match import parse_stack_match
    within = parse_stack_match(within)

    # {{{ change kernel

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, name_gen)
    r_uniq = _ReductionInameUniquifier(
            rule_mapping_context,
            frozenset(_to_inames_tuple(inames)) if inames is not None else inames,
            within=within)

    kernel = rule_mapping_context.finish_kernel(r_uniq.map_kernel(kernel))

    # }}}

    # {{{ duplicate the inames

    from loopy.isl_helpers import duplicate_axes
    from loopy.kernel.tools import DomainChanger

    for old_iname, new_iname in r_uniq.old_to_new:
        domch = DomainChanger(kernel, frozenset([old_iname]))

        kernel = kernel.copy(
                domains=domch.get_domains_with(
                    duplicate_axes(domch.domain, [old_iname], [new_iname])))

    # }}}

    # {{{ copy metadata

    inames = dict(kernel.inames)

    for old_iname, new_iname in r_uniq.old_to_new:
        inames[new_iname] = inames[old_iname].copy(name=new_iname)

    return kernel.copy(inames=inames)

    # }}}


# }}}


# {{{ add_inames_to_insn

@for_each_kernel
def add_inames_to_insn(kernel: LoopKernel,
                       inames: ToInameStrSetConvertible,
                       insn_match: ToMatchConvertible) -> LoopKernel:
    """
    :arg inames: an iterable of inames that will be added to the
        instructions matched by *insn_match*, or a comma-separated
        string that parses to such an iterable.
    :arg insn_match: a match as understood by :func:`loopy.match.parse_match`.

    :returns: a :class:`LoopKernel` with the *inames* added to
        the instructions matched by *insn_match*.

    .. versionadded:: 2016.3
    """

    from loopy.match import parse_match
    match = parse_match(insn_match)
    inames = frozenset(_to_inames_tuple(inames))

    new_instructions: list[InstructionBase] = []
    for insn in kernel.instructions:
        if match(kernel, insn):
            new_instructions.append(
                    insn.copy(within_inames=insn.within_inames | inames))
        else:
            new_instructions.append(insn)

    return kernel.copy(instructions=new_instructions)

# }}}


# {{{ remove_inames_from_insn

@for_each_kernel
def remove_inames_from_insn(
        kernel: LoopKernel,
        inames: ToInameStrSetConvertible,
        insn_match: ToMatchConvertible) -> LoopKernel:
    """Remove inames from kernel instructions.

    This transformation is useful when an iname is added to an
    instruction in a sub-kernel by an inlining call because the
    kernel invocation itself has the iname. When the instruction
    does not depend on the iname, this transformation can be used
    to remove that iname.

    :arg inames: an iterable of inames that will be added to the
        instructions matched by *insn_match*.
    :arg insn_match: a match as understood by :func:`loopy.match.parse_match`.

    :returns: a :class:`LoopKernel` with the *inames* removed from
        the instructions matched by *insn_match*.

    .. versionadded:: 2023.0
    """

    from loopy.match import parse_match
    match = parse_match(insn_match)
    inames = frozenset(_to_inames_tuple(inames))

    new_instructions: list[InstructionBase] = []
    for insn in kernel.instructions:
        if match(kernel, insn):
            new_inames = insn.within_inames - inames
            if new_inames == insn.within_inames:
                raise LoopyError(
                    f"inames {inames} not found in instruction {insn.id}")

            new_instructions.append(insn.copy(within_inames=new_inames))
        else:
            new_instructions.append(insn)

    return kernel.copy(instructions=new_instructions)


@for_each_kernel
def remove_predicates_from_insn(
        kernel: LoopKernel,
        predicates: frozenset[Expression],
        insn_match: ToMatchConvertible) -> LoopKernel:
    """Remove predicates from kernel instructions.

    This transformation is useful when a predicate is added to an
    instruction in a sub-kernel by an inlining call because the
    kernel invocation itself has the iname. When the instruction
    does not depend on the predicate, this transformation can be used
    for removing that predicate.

    :arg predicates: a frozenset of predicates that will be added to the
        instructions matched by *insn_match*
    :arg insn_match: a match as understood by :func:`loopy.match.parse_match`.

    :returns: a :class:`LoopKernel` with the *predicates* removed from
        the instructions matched by *insn_match*.

    .. versionadded:: 2023.0
    """
    if not isinstance(predicates, frozenset):
        raise TypeError(f"'predicates' must be a frozenset: {type(predicates)}")

    from loopy.match import parse_match
    match = parse_match(insn_match)

    new_instructions: list[InstructionBase] = []
    for insn in kernel.instructions:
        if match(kernel, insn):
            new_predicates = insn.predicates - predicates
            new_instructions.append(insn.copy(predicates=frozenset(new_predicates)))
        else:
            new_instructions.append(insn)

    return kernel.copy(instructions=new_instructions)

# }}}


# {{{ map_domain and associated functions

# {{{ _MapDomainMapper

@final
class _MapDomainMapper(RuleAwareIdentityMapper[[]]):
    old_inames: AbstractSet[InameStr]
    new_inames: AbstractSet[InameStr]
    substitutions: Mapping[str, ArithmeticExpression]

    def __init__(self,
                rule_mapping_context: SubstitutionRuleMappingContext,
                new_inames: AbstractSet[InameStr],
                substitutions: Mapping[str, ArithmeticExpression]
            ) -> None:
        super().__init__(rule_mapping_context)

        self.old_inames = frozenset(substitutions)
        self.new_inames = new_inames

        self.substitutions = substitutions

    @override
    def map_reduction(self, expr: Reduction, /,
                      expn_state: ExpansionState) -> Expression:
        red_overlap = frozenset(expr.inames) & self.old_inames
        arg_ctx_overlap = frozenset(expn_state.arg_context) & self.old_inames
        if red_overlap:
            if len(red_overlap) != len(self.old_inames):
                raise LoopyError(
                        f"Reduction '{expr}' involves a part of the map domain "
                        "inames. Reductions must either involve all or none of "
                        "the map domain inames.")

            if arg_ctx_overlap:
                if arg_ctx_overlap == red_overlap:
                    # All variables are shadowed by context, that's OK.
                    return super().map_reduction(
                            expr, expn_state)
                else:
                    raise LoopyError(
                            f"Reduction '{expr}' has some of the reduction "
                            "variables affected by the map_domain shadowed by context. "
                            "Either all or none must be shadowed.")

            new_inames = list(expr.inames)
            for old_iname in self.old_inames:
                new_inames.remove(old_iname)
            new_inames.extend(self.new_inames)

            return Reduction(expr.operation, tuple(new_inames),
                        self.rec(expr.expr, expn_state),
                        expr.allow_simultaneous)
        else:
            return super().map_reduction(expr, expn_state)

    @override
    def map_variable(self, expr: prim.Variable, /,
                     expn_state: ExpansionState) -> Expression:
        if (expr.name in self.old_inames
                and expr.name not in expn_state.arg_context):
            return self.substitutions[expr.name]
        else:
            return super().map_variable(expr, expn_state)

# }}}


# {{{ _apply_identity_for_missing_map_dims(mapping, desired_dims)

def _apply_identity_for_missing_map_dims(
            mapping: isl.BasicMap,
            desired_dims: Sequence[str],
    ) -> tuple[isl.BasicMap, Sequence[tuple[str, str]]]:
    """For every variable *v* in *desired_dims* that is not found in the
    input space for *mapping*, add input dimension *v*, output dimension
    ``v_'proxy'_``, and constraint ``v = v_'proxy'_`` to the mapping.

    :arg mapping: an :class:`islpy.BasicMap`.
    :arg desired_dims: a sequence of :class:`str` specifying the names of the
        desired map input dimensions.

    :returns: a pair containing the mapping with the new dimensions and
        constraints added, and sequence of pairs of :class:`str` values
        specifying the ``(v, v_'proxy'_)`` pairs.
    """

    # If the transform map in map_domain (below) does not contain all the
    # inames in the iname domain (set) to which it is applied, the missing
    # inames must be added to the transform map so that intersect_domain()
    # doesn't remove them from the iname domain when the map is applied.

    # No two map dimension names can match, so we create a unique name for each
    # new variable in the output dimension by appending _'proxy'_, and return a
    # list of the (v, v_'proxy'_) pairs so that the proxy dims can be
    # identified and replaced later.

    # (Apostrophes are not allowed in inames, so this suffix
    # will not match any existing inames. This function is also used on
    # dependency maps, which may contain variable names consisting of an iname
    # suffixed with a single apostrophe.)

    from loopy.isl_helpers import add_and_name_dims, add_eq_constraint_from_names

    # {{{ Find any missing vars and add them to the input and output space

    missing_dims = list(set(desired_dims) - set(mapping.get_var_names(dim_type.in_)))
    augmented_mapping = add_and_name_dims(mapping, dim_type.in_, missing_dims)

    missing_dims_proxies = [f"{d}_'prox'_" for d in missing_dims]
    assert not set(missing_dims_proxies) & set(augmented_mapping.get_var_dict().keys())

    augmented_mapping = add_and_name_dims(
        augmented_mapping, dim_type.out, missing_dims_proxies)

    proxy_name_pairs = list(zip(missing_dims, missing_dims_proxies, strict=True))

    # }}}

    # {{{ Add identity constraint (v = v_'proxy'_) for each new pair of dims

    for real_iname, proxy_iname in proxy_name_pairs:
        augmented_mapping = add_eq_constraint_from_names(
            augmented_mapping, proxy_iname, real_iname)

    # }}}

    return augmented_mapping, proxy_name_pairs

# }}}


# {{{ map_domain

@for_each_kernel
def map_domain(kernel: LoopKernel, transform_map: isl.BasicMap) -> LoopKernel:
    """Transform an iname domain by applying a mapping from existing inames to
    new inames.

    :arg transform_map: a bijective :class:`islpy.BasicMap` from existing inames to
        new inames. To be applicable to a kernel domain, all input inames in
        the map must be found in the domain. The map must be applicable to
        exactly one domain found in *kernel.domains*.
    """

    # FIXME: Express _split_iname_backend in terms of this
    #   Missing/deleted for now:
    #     - slab processing
    #     - priorities processing

    if not transform_map.to_map().is_bijective():
        raise LoopyError("transform_map must be bijective")

    in_var_dict = transform_map.get_var_dict(dim_type.in_)
    transform_map_out_dims = frozenset(transform_map.get_var_dict(dim_type.out))
    transform_map_in_dims = frozenset(in_var_dict)

    # {{{ Make sure that none of the mapped inames are involved in loop priorities

    if hasattr(kernel, "loop_priority") and kernel.loop_priority:
        for prio in kernel.loop_priority:
            if set(prio) & transform_map_in_dims:
                raise ValueError(
                    f"Loop priority {prio!r} contains iname(s) transformed by "
                    f"map {transform_map} in map_domain.")

    # }}}

    # {{{ Solve for representation of old inames in terms of new

    substitutions: dict[str, ArithmeticExpression] = {}
    var_substitutions: dict[prim.Variable, ArithmeticExpression] = {}
    applied_iname_rewrites = kernel.applied_iname_rewrites

    out_vars_in_terms_of_in_vars = transform_map.reverse().to_map().as_pw_multi_aff()

    for iname in transform_map_in_dims:
        subst_from_map = pw_aff_to_expr(
            out_vars_in_terms_of_in_vars.get_pw_aff(in_var_dict[iname][1]))
        substitutions[iname] = subst_from_map
        var_substitutions[prim.Variable(iname)] = subst_from_map

    applied_iname_rewrites = (*applied_iname_rewrites, var_substitutions)
    del var_substitutions

    # }}}

    # {{{ Function to apply mapping to one set

    def process_set(s: isl.BasicSet) -> isl.BasicSet:
        """Return the transformed set. Assume that map is applicable to this
        set."""

        # {{{ Align dims of transform_map and s so that map can be applied

        # Create a map whose input space matches the set
        map_with_s_domain = isl.Map.from_domain(s)

        # {{{ Check for missing map dims and add them

        # For every iname v in the domain that is *not* found in the input
        # space of the transform map, add input dimension v, output dimension
        # v_'proxy'_, and constraint v = v_'proxy'_ to the transform map.
        # Otherwise, v will be dropped from the domain when the map is applied.

        augmented_transform_map, proxy_name_pairs = \
            _apply_identity_for_missing_map_dims(
                transform_map, s.get_var_names_not_none(dim_type.set))

        # }}}

        # {{{ Align transform map input dims with set dims

        # FIXME: Make an exported/documented interface of this in islpy

        dim_types = [dim_type.param, dim_type.in_, dim_type.out]
        # Variables found in iname domain set
        s_names = {
                not_none(map_with_s_domain.get_dim_name(dt, i))
                for dt in dim_types
                for i in range(map_with_s_domain.dim(dt))
                }
        # Variables found in transform map
        map_names = {
                not_none(augmented_transform_map.get_dim_name(dt, i))
                for dt in dim_types
                for i in range(augmented_transform_map.dim(dt))
                }
        # (_align_dim_type uses these two sets to determine which names are in
        # both the obj and template)

        from islpy import _align_dim_type
        aligned_map = _align_dim_type(
                dim_type.param,
                augmented_transform_map, map_with_s_domain, False,
                map_names, s_names)
        aligned_map = _align_dim_type(
                dim_type.in_,
                aligned_map, map_with_s_domain, False,
                map_names, s_names)

        # }}}

        # }}}

        # Apply the transform map to the domain
        new_s = aligned_map.intersect_domain(s).range()

        # Now rename any proxy dims back to their original names

        from loopy.isl_helpers import find_and_rename_dim
        for real_iname, proxy_iname in proxy_name_pairs:
            new_s = find_and_rename_dim(
                new_s, dim_type.set, proxy_iname, real_iname)

        return new_s

        # FIXME: Revive _project_out_only_if_all_instructions_in_within

    # }}}

    # {{{ Apply the transform map to exactly one domain

    map_applied_to_one_dom = False
    new_domains: list[isl.BasicSet] = []
    transform_map_rules = (
        "Transform map must be applicable to exactly one domain. "
        "A transform map is applicable to a domain if its input "
        "inames are a subset of the domain inames.")

    for old_domain in kernel.domains:
        # Make sure transform map is applicable to this set. Then transform.
        if not transform_map_in_dims <= frozenset(old_domain.get_var_dict()):
            # Map not applicable to this set because map transforms at least
            # one iname that is not present in the set. Don't transform.
            new_domains.append(old_domain)
            continue

        elif map_applied_to_one_dom:
            # Map is applicable to this domain, but this map was
            # already applied. Error.
            raise LoopyError(
                f"Transform map {transform_map} was applicable to more than "
                f"one domain. {transform_map_rules}")

        else:
            # Map is applicable to this domain, and this map has not yet
            # been applied. Transform.
            new_domains.append(process_set(old_domain))
            map_applied_to_one_dom = True

    # If we get this far, either the map has been applied to 1 domain (good)
    # or the map could not be applied to any domain, which should produce an error.
    if not map_applied_to_one_dom:
        raise LoopyError(
            f"Transform map {transform_map} was not applicable to any domain. "
            f"{transform_map_rules}")

    # }}}

    # {{{ Update within_inames for each statement

    # If we get this far, we know that the map was applied to exactly one domain,
    # and that all the inames in transform_map_in_dims were transformed to
    # inames in transform_map_out_dims. However, it's still possible that for some
    # statements, stmt.within_inames will contain at least one but not all of the
    # transformed inames (transform_map_in_dims).
    # In this case, it's not clear what within_inames should be. Therefore, we
    # require that if any transformed inames are found in stmt.within_inames,
    # ALL transformed inames must be found in stmt.within_inames.

    new_stmts: list[InstructionBase] = []
    for stmt in kernel.instructions:
        overlap = transform_map_in_dims & stmt.within_inames
        if overlap:
            if len(overlap) != len(transform_map_in_dims):
                raise LoopyError(
                    f"Statement '{stmt.id}' is within only a part of the mapped "
                    f"inames in transformation map {transform_map}. Statements "
                    "must be within all or none of the mapped inames.")

            stmt = stmt.copy(within_inames=(
                stmt.within_inames - transform_map_in_dims) | transform_map_out_dims)
        else:
            # Leave stmt unmodified
            pass

        new_stmts.append(stmt)

    # }}}

    kernel = kernel.copy(
            domains=new_domains,
            instructions=new_stmts,
            applied_iname_rewrites=applied_iname_rewrites)

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    ins = _MapDomainMapper(
            rule_mapping_context, transform_map_out_dims, substitutions)

    kernel = ins.map_kernel(kernel)
    return rule_mapping_context.finish_kernel(kernel)


# }}}

# }}}


@for_each_kernel
def add_inames_for_unused_hw_axes(kernel: LoopKernel,
                                  within: ToMatchConvertible = None) -> LoopKernel:
    """
    Returns a kernel with inames added to each instruction corresponding to any
    hardware-parallel iname tags (:class:`~loopy.kernel.data.GroupInameTag`,
    :class:`~loopy.kernel.data.LocalInameTag`) unused in the instruction but
    used elsewhere in the kernel.

    Current limitations:

    * Only one iname in the kernel may be tagged with each of the unused hw axes.
    * Occurrence of an ``l.auto`` tag when an instruction is missing one of the
      local hw axes.

    :arg within: a match, as understood by :func:`loopy.match.parse_match`.
    """
    from loopy.kernel.data import AutoFitLocalInameTag, GroupInameTag, LocalInameTag

    n_local_axes = max([tag.axis
        for iname in kernel.inames.values()
        for tag in iname.tags
        if isinstance(tag, LocalInameTag)],
        default=-1) + 1

    n_group_axes = max([tag.axis
        for iname in kernel.inames.values()
        for tag in iname.tags
        if isinstance(tag, GroupInameTag)],
        default=-1) + 1

    contains_auto_local_tag = any(isinstance(tag, AutoFitLocalInameTag)
        for iname in kernel.inames.values()
        for tag in iname.tags)

    if contains_auto_local_tag:
        raise LoopyError("kernels containing 'l.auto' tags are invalid arguments")

    # {{{ fill axes_to_inames

    # local_axes_to_inames: ith entry contains the iname tagged with l.i or None
    # if multiple inames are tagged with l.i
    local_axes_to_inames: list[InameStr | None] = []
    # group_axes_to_inames: ith entry contains the iname tagged with g.i or None
    # if multiple inames are tagged with g.i
    group_axes_to_inames: list[InameStr | None] = []

    for i in range(n_local_axes):
        ith_local_axes_tag = LocalInameTag(i)
        inames = [name
                for name, iname in kernel.inames.items()
                if ith_local_axes_tag in iname.tags]
        if not inames:
            raise LoopyError(f"unused local hw axes: {i}")

        local_axes_to_inames.append(inames[0] if len(inames) == 1 else None)

    for i in range(n_group_axes):
        ith_group_axes_tag = GroupInameTag(i)
        inames = [name
                for name, iname in kernel.inames.items()
                if ith_group_axes_tag in iname.tags]
        if not inames:
            raise LoopyError(f"unused group hw axes: {i}")

        group_axes_to_inames.append(inames[0] if len(inames) == 1 else None)

    # }}}

    from loopy.match import parse_match
    within = parse_match(within)

    new_insns: list[InstructionBase] = []

    for insn in kernel.instructions:
        if within(kernel, insn):
            within_tags = frozenset().union(*(kernel.inames[iname].tags
                for iname in insn.within_inames))
            missing_local_axes = [i for i in range(n_local_axes)
                    if LocalInameTag(i) not in within_tags]
            missing_group_axes = [i for i in range(n_group_axes)
                    if GroupInameTag(i) not in within_tags]

            for axis in missing_local_axes:
                iname = local_axes_to_inames[axis]
                if iname:
                    insn = insn.copy(
                        within_inames=insn.within_inames | frozenset([iname]))
                else:
                    raise LoopyError(
                            f"multiple inames tagged with 'l.{axis}' while "
                            f"adding unused local hw axes to instruction '{insn.id}'")

            for axis in missing_group_axes:
                iname = group_axes_to_inames[axis]
                if iname is not None:
                    insn = insn.copy(
                        within_inames=insn.within_inames | frozenset([iname]))
                else:
                    raise LoopyError(
                            f"multiple inames tagged with 'g.{axis}' while "
                            f"adding unused group hw axes to instruction '{insn.id}'")

        new_insns.append(insn)

    return kernel.copy(instructions=new_insns)


# {{{ rename_inames

@for_each_kernel
@remove_any_newly_unused_inames
def rename_inames(
            kernel: LoopKernel,
            old_inames: Collection[InameStr],
            new_iname: InameStr,
            existing_ok: bool = False,
            within: ToStackMatchConvertible = None,
            raise_on_domain_mismatch: bool | None = None
        ) -> LoopKernel:
    r"""
    :arg old_inames: a collection of inames that must be renamed to *new_iname*.
    :arg within: a stack match, as understood by :func:`loopy.match.parse_stack_match`.
    :arg existing_ok: execute even if *new_iname* already exists.
    :arg raise_on_domain_mismatch: if *True*, raises an error if
        :math:`\exists (i_1,i_2) \in \{\text{old\_inames}\}^2 |
        \mathcal{D}_{i_1} \neq \mathcal{D}_{i_2}`.
    """

    if isinstance(old_inames, str) or not isinstance(old_inames, Collection):
        raise LoopyError("'old_inames' must be a collection of strings: "
                         f"{type(old_inames)}")
    old_inames = frozenset(_to_inames_tuple(old_inames))

    if new_iname in old_inames:
        raise LoopyError("'new_iname' is part of inames being renamed: "
                         f"'{new_iname}' in {_to_inames_str(old_inames)}")

    if new_iname in (kinames := (kernel.all_variable_names() - kernel.all_inames())):
        raise LoopyError(f"new iname '{new_iname}' is already a variable in the"
                         f"kernel: {kinames}")

    if any((len(insn.within_inames & old_inames) > 1) for insn in kernel.instructions):
        raise LoopyError("'old_inames' contains nested inames -- renaming is illegal")

    if raise_on_domain_mismatch is None:
        raise_on_domain_mismatch = __debug__

    var_name_gen = kernel.get_var_name_generator()

    # sort to have deterministic implementation.
    sorted_old_inames = sorted(old_inames)

    # FIXME: distinguish existing iname vs. existing other variable
    does_exist = new_iname in kernel.all_inames()

    if not (old_inames <= kernel.all_inames()):
        raise LoopyError(
                f"old inames {_to_inames_str(old_inames - kernel.all_inames())}"
                " do not exist.")

    if does_exist and not existing_ok:
        raise LoopyError(f"iname '{new_iname}' conflicts with an existing identifier"
                         " --cannot rename")

    if not does_exist:
        # {{{ rename old_inames[0] -> new_iname
        # so that the code below can focus on "merging" inames that already exist

        kernel = duplicate_inames(
                kernel, sorted_old_inames[0], within=within, new_inames=[new_iname])

        # old_iname[0] is already renamed to new_iname => do not rename again.
        sorted_old_inames = sorted_old_inames[1:]

        # }}}

    del does_exist
    assert new_iname in kernel.all_inames()

    if raise_on_domain_mismatch:
        for old_iname in sorted_old_inames:
            # {{{ check that the domains match up

            dom = kernel.get_inames_domain(frozenset((old_iname, new_iname)))

            var_dict = dom.get_var_dict()
            _, old_idx = var_dict[old_iname]
            _, new_idx = var_dict[new_iname]

            par_idx = dom.dim(dim_type.param)
            dom_old = dom.move_dims(
                    dim_type.param, par_idx, dim_type.set, old_idx, 1)
            dom_old = dom_old.move_dims(dim_type.set,
                                        dom_old.dim(dim_type.set),
                                        dim_type.param, par_idx, 1)
            dom_old = dom_old.project_out(dim_type.set,
                                          new_idx
                                          if new_idx < old_idx
                                          else new_idx - 1,
                                          1)

            par_idx = dom.dim(dim_type.param)
            dom_new = dom.move_dims(dim_type.param, par_idx,
                                    dim_type.set, new_idx, 1)
            dom_new = dom_new.move_dims(dim_type.set, dom_new.dim(dim_type.set),
                                        dim_type.param, par_idx, 1)
            dom_new = dom_new.project_out(dim_type.set,
                                          old_idx
                                          if old_idx < new_idx
                                          else old_idx - 1, 1)

            if not (dom_old <= dom_new <= dom_old):
                raise LoopyError(
                        f"inames {old_iname} and {new_iname} do not iterate over "
                        "the same domain")

            # }}}

    subst_dict = {old_iname: prim.Variable(new_iname)
                  for old_iname in sorted_old_inames}

    from loopy.match import parse_stack_match
    within = parse_stack_match(within)

    from pymbolic.mapper.substitutor import make_subst_func
    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, var_name_gen)
    smap = RuleAwareSubstitutionMapper(rule_mapping_context,
                    make_subst_func(subst_dict), within)

    from loopy.kernel.instruction import MultiAssignmentBase

    def does_insn_involve_iname(
            kernel: LoopKernel,
            insn: InstructionBase,
            stack: RuleStack) -> bool:
        return bool(not isinstance(insn, MultiAssignmentBase)
                or old_inames & insn.dependency_names()
                or old_inames & insn.reduction_inames())

    kernel = rule_mapping_context.finish_kernel(
            smap.map_kernel(kernel, within=does_insn_involve_iname,
                            map_tvs=False, map_args=False))

    new_instructions = [
        insn.copy(within_inames=(
            (insn.within_inames - old_inames) | frozenset([new_iname])))
        if ((len(old_inames & insn.within_inames) != 0) and within(kernel, insn, ()))
        else insn
        for insn in kernel.instructions]

    kernel = kernel.copy(instructions=new_instructions)

    return kernel


@for_each_kernel
def rename_iname(
            kernel: LoopKernel,
            old_iname: InameStr,
            new_iname: InameStr,
            existing_ok: bool = False,
            within: ToStackMatchConvertible = None,
            preserve_tags: bool = True,
            raise_on_domain_mismatch: bool | None = None
        ) -> LoopKernel:
    r"""
    Single iname version of :func:`loopy.rename_inames`.

    :arg existing_ok: execute even if *new_iname* already exists.
    :arg within: a stack match, as understood by :func:`loopy.match.parse_stack_match`.
    :arg preserve_tags: copy the tags on the old iname to the new iname.
    :arg raise_on_domain_mismatch: If *True*, raises an error if
        :math:`\exists (i_1,i_2) \in \{\text{old\_inames}\}^2 |
        \mathcal{D}_{i_1} \neq \mathcal{D}_{i_2}`.
    """
    from itertools import product

    from loopy import tag_inames

    tags = kernel.inames[old_iname].tags
    kernel = rename_inames(kernel, [old_iname], new_iname, existing_ok,
                           within, raise_on_domain_mismatch)
    if preserve_tags:
        kernel = tag_inames(kernel, list(product([new_iname], tags)))
    return kernel

# }}}


# vim: foldmethod=marker

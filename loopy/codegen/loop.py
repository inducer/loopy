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

from six.moves import range

from loopy.diagnostic import warn, LoopyError
from loopy.codegen.result import merge_codegen_results
import islpy as isl
from islpy import dim_type
from loopy.codegen.control import build_loop_nest
from pymbolic.mapper.stringifier import PREC_NONE


# {{{ conditional-reducing slab decomposition

def get_slab_decomposition(kernel, iname):
    iname_domain = kernel.get_inames_domain(iname)

    if iname_domain.is_empty():
        return ()

    space = iname_domain.space

    lower_incr, upper_incr = kernel.iname_slab_increments.get(iname, (0, 0))
    lower_bulk_bound = None
    upper_bulk_bound = None

    if lower_incr or upper_incr:
        bounds = kernel.get_iname_bounds(iname)

        lower_bound_pw_aff_pieces = bounds.lower_bound_pw_aff.coalesce().get_pieces()
        upper_bound_pw_aff_pieces = bounds.upper_bound_pw_aff.coalesce().get_pieces()

        if len(lower_bound_pw_aff_pieces) > 1:
            raise NotImplementedError("lower bound for slab decomp of '%s' needs "
                    "conditional/has more than one piece" % iname)
        if len(upper_bound_pw_aff_pieces) > 1:
            raise NotImplementedError("upper bound for slab decomp of '%s' needs "
                    "conditional/has more than one piece" % iname)

        (_, lower_bound_aff), = lower_bound_pw_aff_pieces
        (_, upper_bound_aff), = upper_bound_pw_aff_pieces

        from loopy.isl_helpers import iname_rel_aff

        if lower_incr:
            assert lower_incr > 0
            lower_slab = ("initial", isl.BasicSet.universe(space)
                    .add_constraint(
                        isl.Constraint.inequality_from_aff(
                            iname_rel_aff(space,
                                iname, "<", lower_bound_aff+lower_incr))))
            lower_bulk_bound = (
                    isl.Constraint.inequality_from_aff(
                        iname_rel_aff(space,
                            iname, ">=", lower_bound_aff+lower_incr)))
        else:
            lower_slab = None

        if upper_incr:
            assert upper_incr > 0
            upper_slab = ("final", isl.BasicSet.universe(space)
                    .add_constraint(
                        isl.Constraint.inequality_from_aff(
                            iname_rel_aff(space,
                                iname, ">", upper_bound_aff-upper_incr))))
            upper_bulk_bound = (
                    isl.Constraint.inequality_from_aff(
                        iname_rel_aff(space,
                            iname, "<=", upper_bound_aff-upper_incr)))
        else:
            lower_slab = None

        slabs = []

        bulk_slab = isl.BasicSet.universe(space)
        if lower_bulk_bound is not None:
            bulk_slab = bulk_slab.add_constraint(lower_bulk_bound)
        if upper_bulk_bound is not None:
            bulk_slab = bulk_slab.add_constraint(upper_bulk_bound)

        slabs.append(("bulk", bulk_slab))
        if lower_slab:
            slabs.append(lower_slab)
        if upper_slab:
            slabs.append(upper_slab)

        return slabs

    else:
        return [("bulk", (isl.BasicSet.universe(space)))]

# }}}


# {{{ unrolled loops

def generate_unroll_loop(codegen_state, sched_index):
    kernel = codegen_state.kernel

    iname = kernel.schedule[sched_index].iname

    bounds = kernel.get_iname_bounds(iname, constants_only=True)

    from loopy.isl_helpers import (
            static_max_of_pw_aff, static_value_of_pw_aff)
    from loopy.symbolic import pw_aff_to_expr

    length_aff = static_max_of_pw_aff(bounds.size, constants_only=True)

    if not length_aff.is_cst():
        raise LoopyError(
                "length of unrolled loop '%s' is not a constant, "
                "cannot unroll")

    length = int(pw_aff_to_expr(length_aff))

    try:
        lower_bound_aff = static_value_of_pw_aff(
                bounds.lower_bound_pw_aff.coalesce(),
                constants_only=False)
    except Exception as e:
        raise type(e)("while finding lower bound of '%s': " % iname)

    result = []

    for i in range(length):
        idx_aff = lower_bound_aff + i
        new_codegen_state = codegen_state.fix(iname, idx_aff)
        result.append(
                build_loop_nest(new_codegen_state, sched_index+1))

    return merge_codegen_results(codegen_state, result)

# }}}


# {{{ vectorized loops

def generate_vectorize_loop(codegen_state, sched_index):
    kernel = codegen_state.kernel

    iname = kernel.schedule[sched_index].iname

    bounds = kernel.get_iname_bounds(iname, constants_only=True)

    from loopy.isl_helpers import (
            static_max_of_pw_aff, static_value_of_pw_aff)
    from loopy.symbolic import pw_aff_to_expr

    length_aff = static_max_of_pw_aff(bounds.size, constants_only=True)

    if not length_aff.is_cst():
        warn(kernel, "vec_upper_not_const",
                "upper bound for vectorized loop '%s' is not a constant, "
                "cannot vectorize--unrolling instead")
        return generate_unroll_loop(kernel, sched_index, codegen_state)

    length = int(pw_aff_to_expr(length_aff))

    try:
        lower_bound_aff = static_value_of_pw_aff(
                bounds.lower_bound_pw_aff.coalesce(),
                constants_only=False)
    except Exception as e:
        raise type(e)("while finding lower bound of '%s': " % iname)

    if not lower_bound_aff.plain_is_zero():
        warn(kernel, "vec_lower_not_0",
                "lower bound for vectorized loop '%s' is not zero, "
                "cannot vectorize--unrolling instead")
        return generate_unroll_loop(kernel, sched_index, codegen_state)

    # {{{ 'implement' vectorization bounds

    domain = kernel.get_inames_domain(iname)

    from loopy.isl_helpers import make_slab
    slab = make_slab(domain.get_space(), iname,
            lower_bound_aff, lower_bound_aff+length)
    codegen_state = codegen_state.intersect(slab)

    # }}}

    from loopy.codegen import VectorizationInfo
    new_codegen_state = codegen_state.copy(
            vectorization_info=VectorizationInfo(
                iname=iname,
                length=length,
                space=length_aff.space))

    return build_loop_nest(new_codegen_state, sched_index+1)

# }}}


def intersect_kernel_with_slab(kernel, slab, iname):
    from loopy.kernel.tools import DomainChanger

    domch = DomainChanger(kernel, (iname,))
    orig_domain = domch.get_original_domain()
    orig_domain, slab = isl.align_two(orig_domain, slab)
    return domch.get_kernel_with(orig_domain & slab)


# {{{ hw-parallel loop

def set_up_hw_parallel_loops(codegen_state, schedule_index, next_func,
        hw_inames_left=None):
    kernel = codegen_state.kernel

    from loopy.kernel.data import (
            UniqueTag, HardwareParallelTag, LocalIndexTag, GroupIndexTag)

    from loopy.schedule import get_insn_ids_for_block_at
    insn_ids_for_block = get_insn_ids_for_block_at(kernel.schedule, schedule_index)

    if hw_inames_left is None:
        all_inames_by_insns = set()
        for insn_id in insn_ids_for_block:
            all_inames_by_insns |= kernel.insn_inames(insn_id)

        hw_inames_left = [iname
                for iname in all_inames_by_insns
                if isinstance(kernel.iname_to_tag.get(iname), HardwareParallelTag)]

    if not hw_inames_left:
        return next_func(codegen_state)

    global_size, local_size = kernel.get_grid_sizes_for_insn_ids(
            insn_ids_for_block)

    hw_inames_left = hw_inames_left[:]
    iname = hw_inames_left.pop()

    tag = kernel.iname_to_tag.get(iname)

    from loopy.symbolic import GroupHardwareAxisIndex, LocalHardwareAxisIndex

    assert isinstance(tag, UniqueTag)
    if isinstance(tag, GroupIndexTag):
        hw_axis_expr = GroupHardwareAxisIndex(tag.axis)
    elif isinstance(tag, LocalIndexTag):
        hw_axis_expr = LocalHardwareAxisIndex(tag.axis)
    else:
        raise RuntimeError("unexpected hw tag type")

    other_inames_with_same_tag = [
            other_iname for other_iname in kernel.all_inames()
            if isinstance(kernel.iname_to_tag.get(other_iname), UniqueTag)
            and kernel.iname_to_tag.get(other_iname).key == tag.key
            and other_iname != iname]

    # {{{ 'implement' hardware axis boundaries

    if isinstance(tag, LocalIndexTag):
        hw_axis_size = local_size[tag.axis]
    elif isinstance(tag, GroupIndexTag):
        hw_axis_size = global_size[tag.axis]
    else:
        raise RuntimeError("unknown hardware parallel tag")

    result = []

    bounds = kernel.get_iname_bounds(iname)
    domain = kernel.get_inames_domain(iname)

    # It's ok to find a bound that's too "loose". The conditional
    # generators will mop up after us.
    from loopy.isl_helpers import static_min_of_pw_aff
    lower_bound = static_min_of_pw_aff(bounds.lower_bound_pw_aff,
            constants_only=False)

    # These bounds are 'implemented' by the hardware. Make sure
    # that the downstream conditional generators realize that.
    if not isinstance(hw_axis_size, int):
        hw_axis_size, lower_bound = isl.align_two(hw_axis_size, lower_bound)

    from loopy.isl_helpers import make_slab
    slab = make_slab(domain.get_space(), iname,
            lower_bound, lower_bound+hw_axis_size)
    codegen_state = codegen_state.intersect(slab)

    from loopy.symbolic import pw_aff_to_expr
    hw_axis_expr = hw_axis_expr + pw_aff_to_expr(lower_bound)

    # }}}

    slabs = get_slab_decomposition(kernel, iname)

    if other_inames_with_same_tag and len(slabs) > 1:
        raise RuntimeError("cannot do slab decomposition on inames that share "
                "a tag with other inames")

    result = []

    for slab_name, slab in slabs:
        if len(slabs) > 1:
            result.append(
                    codegen_state.ast_builder.emit_comment(
                        "%s slab for '%s'" % (slab_name, iname)))

        # Have the conditional infrastructure generate the
        # slabbing conditionals.
        slabbed_kernel = intersect_kernel_with_slab(kernel, slab, iname)
        new_codegen_state = (codegen_state
                .copy_and_assign(iname, hw_axis_expr)
                .copy(kernel=slabbed_kernel))

        inner = set_up_hw_parallel_loops(
                new_codegen_state, schedule_index, next_func,
                hw_inames_left)

        result.append(inner)

    return merge_codegen_results(codegen_state, result)

# }}}


# {{{ sequential loop

def generate_sequential_loop_dim_code(codegen_state, sched_index):
    kernel = codegen_state.kernel

    ecm = codegen_state.expression_to_code_mapper
    loop_iname = kernel.schedule[sched_index].iname

    slabs = get_slab_decomposition(kernel, loop_iname)

    from loopy.codegen.bounds import get_usable_inames_for_conditional

    # Note: this does not include loop_iname itself!
    usable_inames = get_usable_inames_for_conditional(kernel, sched_index)
    domain = kernel.get_inames_domain(loop_iname)

    result = []

    for slab_name, slab in slabs:
        cmt = "%s slab for '%s'" % (slab_name, loop_iname)
        if len(slabs) == 1:
            cmt = None

        # {{{ find bounds

        aligned_domain = isl.align_spaces(domain, slab, across_dim_types=True,
                obj_bigger_ok=True)

        dom_and_slab = aligned_domain & slab

        assumptions_non_param = isl.BasicSet.from_params(kernel.assumptions)
        dom_and_slab, assumptions_non_param = isl.align_two(
                dom_and_slab, assumptions_non_param)
        dom_and_slab = dom_and_slab & assumptions_non_param

        # move inames that are usable into parameters
        moved_inames = []
        for iname in dom_and_slab.get_var_names(dim_type.set):
            if iname in usable_inames:
                moved_inames.append(iname)
                dt, idx = dom_and_slab.get_var_dict()[iname]
                dom_and_slab = dom_and_slab.move_dims(
                        dim_type.param, dom_and_slab.dim(dim_type.param),
                        dt, idx, 1)

        _, loop_iname_idx = dom_and_slab.get_var_dict()[loop_iname]

        from loopy.isl_helpers import (
                static_min_of_pw_aff,
                static_max_of_pw_aff)

        lbound = (
                kernel.cache_manager.dim_min(
                    dom_and_slab, loop_iname_idx)
                .gist(kernel.assumptions)
                .coalesce())
        ubound = (
            kernel.cache_manager.dim_max(
                dom_and_slab, loop_iname_idx)
            .gist(kernel.assumptions)
            .coalesce())

        static_lbound = static_min_of_pw_aff(
                lbound,
                constants_only=False)
        static_ubound = static_max_of_pw_aff(
                ubound,
                constants_only=False)

        # }}}

        # {{{ find implemented slab, build inner code

        from loopy.isl_helpers import make_slab_from_bound_pwaffs

        # impl_slab may be overapproximated
        impl_slab = make_slab_from_bound_pwaffs(
                dom_and_slab.space,
                loop_iname, static_lbound, static_ubound)

        for iname in moved_inames:
            dt, idx = impl_slab.get_var_dict()[iname]
            impl_slab = impl_slab.move_dims(
                    dim_type.set, impl_slab.dim(dim_type.set),
                    dt, idx, 1)

        new_codegen_state = (
                codegen_state
                .intersect(impl_slab)
                .copy(kernel=intersect_kernel_with_slab(
                    kernel, slab, iname)))

        inner = build_loop_nest(new_codegen_state, sched_index+1)

        # }}}

        if cmt is not None:
            result.append(codegen_state.ast_builder.emit_comment(cmt))

        from loopy.symbolic import aff_to_expr

        astb = codegen_state.ast_builder

        if (static_ubound - static_lbound).plain_is_zero():
            # single-trip, generate just a variable assignment, not a loop
            result.append(merge_codegen_results(codegen_state, [
                astb.emit_initializer(
                    codegen_state,
                    kernel.index_dtype, loop_iname,
                    ecm(aff_to_expr(static_lbound), PREC_NONE, "i"),
                    is_const=True),
                astb.emit_blank_line(),
                inner,
                ]))

        else:
            inner_ast = inner.current_ast(codegen_state)
            result.append(
                inner.with_new_ast(
                    codegen_state,
                    astb.emit_sequential_loop(
                        codegen_state, loop_iname, kernel.index_dtype,
                        static_lbound, static_ubound, inner_ast)))

    return merge_codegen_results(codegen_state, result)

# }}}

# vim: foldmethod=marker

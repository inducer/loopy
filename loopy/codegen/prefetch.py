from __future__ import division

from pytools import Record
import pyopencl as cl
import pyopencl.characterize as cl_char
from loopy.codegen import wrap_in, gen_code_block
import islpy as isl
from islpy import dim_type
import numpy as np




# {{{ prefetch preprocessing

def preprocess_prefetch(kernel):
    """Assign names, dim storage lengths to prefetches.
    """

    all_pf_list = kernel.prefetch.values()
    new_prefetch_dict = {}
    lmem_size = cl_char.usable_local_mem_size(kernel.device)

    for i_pf, pf in enumerate(kernel.prefetch.itervalues()):
        all_pf_nbytes = [opf.nbytes for opf in all_pf_list]
        other_pf_sizes = sum(all_pf_nbytes[:i_pf]+all_pf_nbytes[i_pf+1:])

        dim_storage_lengths = [stop-start for start, stop in
                [pf.dim_bounds_by_iname[iname] for iname in pf.all_inames()]]

        # sizes of all dims except the last one, which we may change
        # below to avoid bank conflicts
        from pytools import product
        other_dim_sizes = (pf.itemsize
                * product(dim_storage_lengths[:-1]))

        if kernel.device.local_mem_type == cl.device_local_mem_type.GLOBAL:
            # FIXME: could try to avoid cache associativity disasters
            new_dsl = dim_storage_lengths

        elif kernel.device.local_mem_type == cl.device_local_mem_type.LOCAL:
            min_mult = cl_char.local_memory_bank_count(kernel.device)
            good_incr = None
            new_dsl = dim_storage_lengths
            min_why_not = None

            for increment in range(dim_storage_lengths[-1]//2):

                test_dsl = dim_storage_lengths[:]
                test_dsl[-1] = test_dsl[-1] + increment
                new_mult, why_not = cl_char.why_not_local_access_conflict_free(
                        kernel.device, pf.itemsize,
                        pf.dim_lengths(), test_dsl)

                # will choose smallest increment 'automatically'
                if new_mult < min_mult:
                    new_lmem_use = other_pf_sizes + pf.itemsize*product(new_dsl)
                    if new_lmem_use < lmem_size:
                        new_dsl = test_dsl
                        min_mult = new_mult
                        min_why_not = why_not
                        good_incr = increment

            if min_mult != 1:
                from warnings import warn
                from loopy import LoopyAdvisory
                warn("could not find a conflict-free mem layout "
                        "for prefetch of '%s' "
                        "(currently: %dx conflict, increment: %d, reason: %s)"
                        % (pf.input_vector, min_mult, good_incr, min_why_not),
                        LoopyAdvisory)
        else:
            from warnings import warn
            warn("unknown type of local memory")

            new_dsl = dim_storage_lengths


        new_pf = pf.copy(dim_storage_lengths=new_dsl,
                name="prefetch_%s_%d" % (pf.input_vector, i_pf))
        new_prefetch_dict[pf.input_vector, pf.index_expr] = new_pf
        all_pf_list[i_pf] = new_pf

    return kernel.copy(prefetch=new_prefetch_dict)

# }}}

# {{{ lmem prefetch code generation

class FetchLoopNestData(Record):
    pass

def make_fetch_loop_nest(flnd, fetch_dim_idx, pf_dim_exprs, iname_subst_map,
        implemented_domain):
    pf = flnd.prefetch
    ccm = flnd.c_code_mapper
    no_pf_ccm = flnd.no_prefetch_c_code_mapper
    kernel = flnd.kernel

    from pymbolic import var
    from cgen import Assign, For, If

    from pymbolic.mapper.substitutor import substitute
    if fetch_dim_idx >= len(pf.fetch_dims):
        # done, return
        from pymbolic.primitives import Variable, Subscript

        from pymbolic.mapper.stringifier import PREC_NONE
        result = Assign(
                pf.name + "".join("[%s]" % ccm(dexpr)
                    for dexpr in pf_dim_exprs),
                no_pf_ccm(
                    Subscript(
                        Variable(pf.input_vector),
                        substitute(pf.index_expr, iname_subst_map)),
                    PREC_NONE))

        def my_ccm(expr):
            return ccm(substitute(expr, iname_subst_map))

        from pymbolic.mapper.dependency import DependencyMapper
        check_vars = [v.name for v in DependencyMapper()(pf.index_expr)]

        from loopy.codegen.bounds import wrap_in_bounds_checks
        return wrap_in_bounds_checks(my_ccm, pf.kernel.domain,
                check_vars, implemented_domain, result)

    fetch_inames = pf.fetch_dims[fetch_dim_idx]
    realiz_inames = flnd.realization_inames[fetch_dim_idx]

    fetch_iname_lengths = [stop-start
            for start, stop in 
            [pf.dim_bounds_by_iname[iname] for iname in fetch_inames]]

    from pytools import product
    dim_length = product(fetch_iname_lengths)

    idx_var_name = "loopy_prefetch_dim_idx_%d" % fetch_dim_idx
    idx_var = var(idx_var_name)

    if realiz_inames is not None:
        # {{{ parallel fetch

        # {{{ find strides per fetch iname

        fetch_iname_strides = [1]
        for fil in fetch_iname_lengths[:0:-1]:
            fetch_iname_strides.insert(0,
                    fetch_iname_strides[0]*fil)

        # }}}

        idx_var_expr_from_inames = sum(stride*var(iname)
                for iname, stride in zip(fetch_inames, fetch_iname_strides))

        # {{{ find expressions for each iname from idx_var

        pf_dim_exprs = pf_dim_exprs[:]
        iname_subst_map = iname_subst_map.copy()

        for i, iname in enumerate(fetch_inames):
            iname_lower, iname_upper = pf.dim_bounds_by_iname[iname]
            iname_len = iname_upper-iname_lower
            iname_val_base = (idx_var // fetch_iname_strides[i])
            if i != 0:
                # the outermost iname is the 'largest', no need to
                # 'modulo away' any larger ones
                iname_val_base = iname_val_base % iname_len

            pf_dim_exprs.append(iname_val_base)
            iname_subst_map[iname] = iname_val_base + iname_lower

        # }}}

        # {{{ build an implemented domain with an extra index variable

        from loopy.symbolic import eq_constraint_from_expr
        idx_var_dim_idx = implemented_domain.get_dim().size(dim_type.set)
        impl_domain_with_index_var = implemented_domain.add_dims(dim_type.set, 1)
        impl_domain_with_index_var = (
                impl_domain_with_index_var
                .set_dim_name(dim_type.set, idx_var_dim_idx, idx_var_name))
        aug_space = impl_domain_with_index_var.get_dim()
        impl_domain_with_index_var = (
                impl_domain_with_index_var
                .intersect(
                    isl.Set.universe(aug_space)
                    .add_constraint(
                        eq_constraint_from_expr(
                            aug_space,
                            idx_var_expr_from_inames - idx_var))))

        # }}}

        realiz_bounds = [
                flnd.kernel.get_bounds(rn, (rn,), allow_parameters=False)
                for rn in realiz_inames]
        for realiz_start, realiz_stop, realiz_equality in realiz_bounds:
            assert not realiz_equality

        realiz_lengths = [stop-start for start, stop, equality in realiz_bounds]
        from pytools import product
        total_realiz_size = product(realiz_lengths)

        result = []

        cur_index = 0

        while cur_index < dim_length:
            pf_idx_expr = 0
            for realiz_iname, length in zip(realiz_inames, realiz_lengths):
                tag = flnd.kernel.iname_to_tag[realiz_iname]
                from loopy.kernel import TAG_WORK_ITEM_IDX
                assert isinstance(tag, TAG_WORK_ITEM_IDX)

                pf_idx_expr = (pf_idx_expr*length
                        + var("(int) get_local_id(%d)" % tag.axis))

            pf_idx_expr += cur_index

            from loopy.isl import make_slab
            new_impl_domain = (
                    impl_domain_with_index_var
                    .intersect(
                        make_slab(
                            impl_domain_with_index_var.get_dim(), idx_var_name,
                            cur_index,
                            min(dim_length, cur_index+total_realiz_size)))
                    .project_out(dim_type.set, idx_var_dim_idx, 1))

            inner = make_fetch_loop_nest(flnd, fetch_dim_idx+1,
                    pf_dim_exprs, iname_subst_map,
                    new_impl_domain)

            if cur_index+total_realiz_size > dim_length:
                inner = wrap_in(If,
                        "%s < %s" % (idx_var_name, dim_length),
                        inner)

            from cgen import Initializer, Const, POD
            inner = gen_code_block([
                Initializer(Const(POD(np.int32, idx_var_name)),
                    ccm(pf_idx_expr)),
                inner], denest=True)

            result.append(inner)

            cur_index += total_realiz_size

        return gen_code_block(result)

        # }}}
    else:
        # {{{ sequential fetch

        if len(fetch_inames) > 1:
            raise NotImplementedError("merged sequential fetches are not supported")
        pf_iname, = fetch_inames

        lb_cns, ub_cns = pf.get_dim_bounds_constraints_by_iname(pf_iname)

        from loopy.isl import cast_constraint_to_space
        loop_slab = (isl.Set.universe(flnd.kernel.space)
                .add_constraints([cast_constraint_to_space(cns, kernel.space)
                    for cns in [lb_cns, ub_cns]]))
        new_impl_domain = implemented_domain.intersect(loop_slab)

        iname_subst_map = iname_subst_map.copy()
        iname_subst_map[pf_iname] = idx_var + pf.dim_bounds_by_iname[pf_iname][0]
        inner = make_fetch_loop_nest(flnd, fetch_dim_idx+1,
                pf_dim_exprs+[idx_var], iname_subst_map,
                new_impl_domain)

        return wrap_in(For,
                "int %s = 0" % idx_var_name,
                "%s < %s" % (idx_var_name, ccm(dim_length)),
                "++%s" % idx_var_name,
                inner)

        # }}}


def generate_prefetch_code(cgs, kernel, sched_index, exec_domain):
    implemented_domain = exec_domain.implemented_domain

    from cgen import Statement as S, Line, Comment

    ccm = cgs.c_code_mapper

    scheduled_pf = kernel.schedule[sched_index]
    pf = kernel.prefetch[
            scheduled_pf.input_vector, scheduled_pf.index_expr]

    # Prefetch has a good amount of flexibility over what axes it
    # uses to accomplish the prefetch. In particular, it can (and should!)
    # use all work group dimensions.

    # {{{ determine which loop axes are used to realize the fetch

    # realization_dims is a list of lists of inames, to represent when two dims jointly
    # make up one fetch axis

    realization_inames = [None] * len(pf.fetch_dims)

    # {{{ first, fix the user-specified fetch dims

    from loopy.kernel import TAG_WORK_ITEM_IDX
    knl_work_item_inames = kernel.ordered_inames_by_tag_type(TAG_WORK_ITEM_IDX)

    for realization_dim_idx, loc_fetch_axis_list in \
            getattr(pf, "loc_fetch_axes", {}).iteritems():
        realization_inames[realization_dim_idx] = [knl_work_item_inames.pop(axis)
            for axis in loc_fetch_axis_list]

    # }}}

    # {{{ next use the work group dimensions, least-stride dim first

    from loopy.kernel import ImageArg, ScalarArg
    from loopy.symbolic import CoefficientCollector

    index_expr = pf.index_expr
    if not isinstance(index_expr, tuple):
        index_expr = (index_expr,)

    arg = kernel.arg_dict[pf.input_vector]
    if isinstance(arg, ImageArg):
        # arbitrary
        ary_strides = (1, 1, 1)[:arg.dimensions]
    else:
        ary_strides = arg.strides
        if ary_strides is None and len(index_expr) == 1:
            ary_strides = (1,)

    iname_to_stride = {}
    for iexpr_i, stride in zip(index_expr, ary_strides):
        coeffs = CoefficientCollector()(iexpr_i)
        for var_name, coeff in coeffs.iteritems():
            if var_name != 1:
                new_stride = coeff*stride
                old_stride = iname_to_stride.get(var_name, None)
                if old_stride is None or new_stride < old_stride:
                    iname_to_stride[var_name] = new_stride

    approximate_arg_values = dict(
            (arg.name, arg.approximately)
            for arg in kernel.args
            if isinstance(arg, ScalarArg))

    def stride_key(fetch_dim_idx):
        fetch_dim = pf.fetch_dims[fetch_dim_idx]

        from pymbolic import evaluate
        key = min(
                evaluate(iname_to_stride[iname], approximate_arg_values)
                for iname in fetch_dim)
        assert isinstance(key, int)
        return key

    pf_fetch_dim_strides = sorted((dim_idx
        for dim_idx in range(len(pf.fetch_dims))
        if realization_inames[dim_idx] is None),
        key=stride_key)

    while knl_work_item_inames and pf_fetch_dim_strides:
        # grab least-stride prefetch dim
        least_stride_pf_fetch_dim_idx = pf_fetch_dim_strides.pop(0)

        # FIXME: It might be good to join multiple things together here
        # for size reasons
        realization_inames[least_stride_pf_fetch_dim_idx] \
                = [knl_work_item_inames.pop(0)]

    if knl_work_item_inames:
        # FIXME
        from warnings import warn
        warn("There were leftover work group dimensions in prefetch "
                "assignment. For now, this won't lead to wrong code, "
                "but it will lead to unnecessary memory bandwidth use.")

    # }}}

    # }}}

    # {{{ generate fetch code

    from loopy.codegen.bounds import get_valid_check_vars
    valid_index_vars = get_valid_check_vars(kernel, sched_index,
            allow_ilp=True,
            exclude_tag_classes=(TAG_WORK_ITEM_IDX,))

    from loopy.symbolic import LoopyCCodeMapper
    flnd = FetchLoopNestData(prefetch=pf,
            no_prefetch_c_code_mapper=
            LoopyCCodeMapper(kernel, no_prefetch=True),
            c_code_mapper=ccm,
            realization_inames=realization_inames,
            kernel=kernel,
            valid_index_vars=valid_index_vars)

    fetch_block = make_fetch_loop_nest(flnd, 0, [], {}, implemented_domain)

    # }}}

    new_block = []

    # {{{ generate comments explaining dimension mapping

    new_block.append(Comment("prefetch %s -- using dimension mapping:" % pf.input_vector))
    for iaxis, (fetch_dim, realiz_inames) in enumerate(zip(pf.fetch_dims, realization_inames)):
        new_block.append(Comment("  fetch axis %d:" % iaxis))
        for iname in fetch_dim:
            iname_lwr, iname_upr = pf.dim_bounds_by_iname[iname]
            new_block.append(Comment("      %s [%d..%d)" % (iname, iname_lwr, iname_upr)))
        new_block.append(Comment("    using:"))

        if realiz_inames is None:
            new_block.append(Comment("      loop"))
        else:
            for realiz_iname in realiz_inames:
                rd_iname_descr = "loop"
                iname_lwr, iname_upr, iname_eq = flnd.kernel.get_bounds(realiz_iname, (realiz_iname,),
                        allow_parameters=False)
                assert not iname_eq

                new_block.append(Comment("      %s (%s) [%s..%s)"
                    % (realiz_iname, kernel.iname_to_tag[realiz_iname],
                        iname_lwr, iname_upr)))

    new_block.append(Line())

    # }}}

    # {{{ omit head sync primitive if possible

    head_sync_unneeded_because = None

    from loopy.prefetch import LocalMemoryPrefetch
    if (sched_index-1 >= 0 
            and isinstance(kernel.schedule[sched_index-1], LocalMemoryPrefetch)):
        head_sync_unneeded_because = "next outer schedule item is a prefetch"

    from pytools import all
    from loopy.kernel import ParallelTag
    from loopy.schedule import ScheduledLoop
    outer_tags = [
            kernel.iname_to_tag.get(sched_item.iname)
            for sched_item in kernel.schedule[:sched_index]
            if isinstance(sched_item, ScheduledLoop)]

    if not [tag
            for tag in outer_tags
            if not isinstance(tag, ParallelTag)]:
        head_sync_unneeded_because = "no sequential axes nested around fetch"

    # generate (no) head sync code
    if head_sync_unneeded_because is None:
        new_block.append(S("barrier(CLK_LOCAL_MEM_FENCE)"))
    else:
        new_block.append(Comment("no sync needed: " + head_sync_unneeded_because))
        new_block.append(Line())

    # }}}

    new_block.append(fetch_block)

    # {{{ omit tail sync primitive if possible

    tail_sync_unneeded_because = None

    if (sched_index+1 < len(kernel.schedule)
            and isinstance(kernel.schedule[sched_index+1], LocalMemoryPrefetch)):
        tail_sync_unneeded_because = "next inner schedule item is a prefetch"

    if tail_sync_unneeded_because is None:
        new_block.append(S("barrier(CLK_LOCAL_MEM_FENCE)"))
    else:
        new_block.append(Line())
        new_block.append(Comment("no sync needed: " + tail_sync_unneeded_because))

    # }}}

    from loopy.codegen.dispatch import build_loop_nest
    new_block.extend([Line(),
        build_loop_nest(cgs, kernel, sched_index+1, exec_domain)])

    return gen_code_block(new_block)

# }}}

# vim: foldmethod=marker

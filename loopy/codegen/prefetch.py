from __future__ import division

from pytools import Record
import pyopencl as cl
import pyopencl.characterize as cl_char
from loopy.codegen import wrap_in, gen_code_block




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

        shape = [stop-start for start, stop in pf.dim_bounds]
        dim_storage_lengths = shape[:]

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
                        shape, test_dsl)

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

def make_fetch_loop_nest(flnd, pf_iname_idx, pf_dim_exprs, pf_idx_subst_map,
        implemented_domain):
    pf = flnd.prefetch
    ccm = flnd.c_code_mapper
    no_pf_ccm = flnd.no_prefetch_c_code_mapper
    kernel = flnd.kernel

    from pymbolic import var
    from cgen import Assign, For, If

    from pymbolic.mapper.substitutor import substitute
    if pf_iname_idx >= len(pf.inames):
        # done, return
        from pymbolic.primitives import Variable, Subscript

        from pymbolic.mapper.stringifier import PREC_NONE
        result = Assign(
                pf.name + "".join("[%s]" % ccm(dexpr)
                    for dexpr in pf_dim_exprs),
                no_pf_ccm(
                    Subscript(
                        Variable(pf.input_vector),
                        substitute(pf.index_expr, pf_idx_subst_map)),
                    PREC_NONE))

        def my_ccm(expr):
            return ccm(substitute(expr, pf_idx_subst_map))

        from pymbolic.mapper.dependency import DependencyMapper
        check_vars = [v.name for v in DependencyMapper()(pf.index_expr)]

        from loopy.codegen.bounds import wrap_in_bounds_checks
        return wrap_in_bounds_checks(my_ccm, pf.kernel.domain,
                check_vars, implemented_domain, result)

    pf_iname = pf.inames[pf_iname_idx]
    realiz_inames = flnd.realization_inames[pf_iname_idx]

    start_index, stop_index = flnd.kernel.get_projected_bounds(pf_iname)
    try:
        start_index = int(start_index)
        stop_index = int(stop_index)
    except TypeError:
        raise RuntimeError("loop bounds for prefetch must be "
                "known statically at code gen time")

    dim_length = stop_index-start_index

    if realiz_inames is not None:
        # {{{ parallel fetch

        realiz_bounds = [flnd.kernel.get_projected_bounds(rn) for rn in realiz_inames]
        realiz_lengths = [stop-start for start, stop in realiz_bounds]
        from pytools import product
        total_realiz_size = product(realiz_lengths)

        result = []

        cur_index = 0

        while start_index+cur_index < stop_index:
            pf_dim_expr = 0
            for realiz_iname, length in zip(realiz_inames, realiz_lengths):
                tag = flnd.kernel.iname_to_tag[realiz_iname]
                from loopy.kernel import TAG_WORK_ITEM_IDX
                assert isinstance(tag, TAG_WORK_ITEM_IDX)

                pf_dim_expr = (pf_dim_expr*length
                        + var("(int) get_local_id(%d)" % tag.axis))

            from loopy.isl import make_slab
            loop_slab = make_slab(pf.kernel.space, pf_iname,
                    start_index+cur_index,
                    min(stop_index, start_index+cur_index+total_realiz_size))
            new_impl_domain = implemented_domain.intersect(loop_slab)

            pf_dim_expr += cur_index

            pf_idx_subst_map = pf_idx_subst_map.copy()
            pf_idx_subst_map[pf_iname] = pf_dim_expr + start_index
            inner = make_fetch_loop_nest(flnd, pf_iname_idx+1,
                    pf_dim_exprs+[pf_dim_expr], pf_idx_subst_map,
                    new_impl_domain)

            if cur_index+total_realiz_size > dim_length:
                inner = wrap_in(If,
                        "%s < %s" % (ccm(pf_dim_expr), stop_index),
                        inner)

            result.append(inner)

            cur_index += total_realiz_size

        return gen_code_block(result)

        # }}}
    else:
        # {{{ sequential fetch

        pf_dim_var = "prefetch_dim_idx_%d" % pf_iname_idx
        pf_dim_expr = var(pf_dim_var)

        lb_cns, ub_cns = flnd.kernel.get_projected_bounds_constraints(pf_iname)
        import islpy as isl
        from loopy.isl import cast_constraint_to_space
        loop_slab = (isl.Set.universe(flnd.kernel.space)
                .add_constraint(cast_constraint_to_space(lb_cns, kernel.space))
                .add_constraint(cast_constraint_to_space(ub_cns, kernel.space)))
        new_impl_domain = implemented_domain.intersect(loop_slab)

        pf_idx_subst_map = pf_idx_subst_map.copy()
        pf_idx_subst_map[pf_iname] = pf_dim_expr + start_index
        inner = make_fetch_loop_nest(flnd, pf_iname_idx+1,
                pf_dim_exprs+[pf_dim_expr], pf_idx_subst_map,
                new_impl_domain)

        return wrap_in(For,
                "int %s = 0" % pf_dim_var,
                "%s < %s" % (pf_dim_var, ccm(dim_length)),
                "++%s" % pf_dim_var,
                inner)

        # }}}


def generate_prefetch_code(cgs, kernel, sched_index, exec_domain):
    implemented_domain = exec_domain.implemented_domain

    from cgen import Statement as S, Line, Comment

    ccm = cgs.c_code_mapper

    # find surrounding schedule items
    if sched_index-1 >= 0:
        next_outer_sched_item = kernel.schedule[sched_index-1]
    else:
        next_outer_sched_item = None

    if sched_index+1 < len(kernel.schedule):
        next_inner_sched_item = kernel.schedule[sched_index+1]
    else:
        next_inner_sched_item = None

    scheduled_pf = kernel.schedule[sched_index]
    pf = kernel.prefetch[
            scheduled_pf.input_vector, scheduled_pf.index_expr]

    # Prefetch has a good amount of flexibility over what axes it
    # uses to accomplish the prefetch. In particular, it can (and should!)
    # use all work group dimensions.

    # {{{ determine which loop axes are used to realize the fetch

    # realization_dims is a list of lists of inames, to represent when two dims jointly
    # make up one fetch axis

    realization_inames = [None] * len(pf.inames)

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

    def stride_key(iname):
        iname_stride = iname_to_stride[iname]

        from pymbolic import evaluate
        key = evaluate(iname_stride, approximate_arg_values)
        assert isinstance(key, int)
        return key

    pf_iname_strides = sorted((iname
        for dim_idx, iname in enumerate(pf.inames)
        if realization_inames[dim_idx] is None),
        key=stride_key)

    while knl_work_item_inames and pf_iname_strides:
        # grab least-stride prefetch dim
        least_stride_pf_iname = pf_iname_strides.pop(0)

        # FIXME: It might be good to join multiple things together here
        # for size reasons
        realization_inames[pf.inames.index(least_stride_pf_iname)] \
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

    from loopy.schedule import get_valid_index_vars
    valid_index_vars = get_valid_index_vars(kernel, sched_index,
            exclude_tags=(TAG_WORK_ITEM_IDX,))

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

    new_block = [
            Comment("prefetch %s[%s] using %s" % (
                pf.input_vector,
                ", ".join(pf.inames),
                ", ".join(
                        (" x ".join("%s(%s)" % (realiz_iname, kernel.iname_to_tag[realiz_iname])
                        for realiz_iname in realiz_inames)
                        if realiz_inames is not None else "loop")
                        for realiz_inames in realization_inames))),
            Line(),
            ]

    # omit head sync primitive if we just came out of a prefetch

    from loopy.prefetch import LocalMemoryPrefetch
    if not isinstance(next_outer_sched_item, LocalMemoryPrefetch):
        new_block.append(S("barrier(CLK_LOCAL_MEM_FENCE)"))
    else:
        new_block.append(Comment("next outer schedule item is a prefetch: "
            "no sync needed"))

    new_block.extend([
        fetch_block,
        ])

    # omit tail sync primitive if we're headed into another prefetch
    if not isinstance(next_inner_sched_item, LocalMemoryPrefetch):
        new_block.append(S("barrier(CLK_LOCAL_MEM_FENCE)"))
    else:
        new_block.append(Line())
        new_block.append(Comment("next inner schedule item is a prefetch: "
            "no sync needed"))

    from loopy.codegen.dispatch import build_loop_nest
    new_block.extend([Line(),
        build_loop_nest(cgs, kernel, sched_index+1, exec_domain)])

    return gen_code_block(new_block)

# }}}

from __future__ import division

import islpy as isl
from islpy import dim_type
from loopy.symbolic import get_dependencies, SubstitutionMapper
from pymbolic.mapper.substitutor import make_subst_func
import numpy as np

from pytools import Record
from pymbolic import var




class InvocationDescriptor(Record):
    __slots__ = ["expr", "args", ]




def to_parameters_or_project_out(param_inames, set_inames, set):
    for iname in set.get_space().get_var_dict().keys():
        if iname in param_inames:
            dt, idx = set.get_space().get_var_dict()[iname]
            set = set.move_dims(
                    dim_type.param, set.dim(dim_type.param),
                    dt, idx, 1)
        elif iname in set_inames:
            pass
        else:
            dt, idx = set.get_space().get_var_dict()[iname]
            set = set.project_out(dt, idx, 1)

    return set




def get_footprint(kernel, subst_name, old_arg_names, arg_names,
        sweep_inames, invocation_descriptors):
    global_footprint_map = None

    processed_sweep_inames = set()

    for invdesc in invocation_descriptors:

        for iname in sweep_inames:
            if iname in old_arg_names:
                arg_idx = old_arg_names.index(iname)
                processed_sweep_inames.add(
                        get_dependencies(invdesc.args[arg_idx]))
            else:
                processed_sweep_inames.add(iname)

        # {{{ construct, check mapping

        map_space = kernel.space
        ln = len(arg_names)
        rn = kernel.space.dim(dim_type.out)

        map_space = map_space.add_dims(dim_type.in_, ln)
        for i, iname in enumerate(arg_names):
            map_space = map_space.set_dim_name(dim_type.in_, i, iname+"'")

        set_space = map_space.move_dims(
                dim_type.out, rn,
                dim_type.in_, 0, ln).range()

        footprint_map = None

        from loopy.symbolic import aff_from_expr
        for uarg_name, arg_val in zip(arg_names, invdesc.args):
            cns = isl.Constraint.equality_from_aff(
                    aff_from_expr(set_space, var(uarg_name+"'") - arg_val))

            cns_map = isl.BasicMap.from_constraint(cns)
            if footprint_map is None:
                footprint_map = cns_map
            else:
                footprint_map = footprint_map.intersect(cns_map)

        footprint_map = footprint_map.move_dims(
                dim_type.in_, 0,
                dim_type.out, rn, ln)

        if global_footprint_map is None:
            global_footprint_map = footprint_map
        else:
            global_footprint_map = global_footprint_map.union(footprint_map)

        # }}}

    processed_sweep_inames = list(processed_sweep_inames)

    global_footprint_map = (isl.Map.from_basic_map(global_footprint_map)
            .intersect_range(kernel.domain))

    # move non-sweep-dimensions into parameter space
    sweep_footprint_map = global_footprint_map.coalesce()

    for iname in kernel.all_inames():
        if iname not in processed_sweep_inames:
            sp = sweep_footprint_map.get_space()
            dt, idx = sp.get_var_dict()[iname]
            sweep_footprint_map = sweep_footprint_map.move_dims(
                    dim_type.param, sp.dim(dim_type.param),
                    dt, idx, 1)

    # compute bounding boxes to each set of parameters
    sfm_dom = sweep_footprint_map.domain().coalesce()

    if not sfm_dom.is_bounded():
        raise RuntimeError("In precomputation of substitution '%s': "
                "sweep did not result in a bounded footprint"
                % subst_name)

    from loopy.kernel import find_var_base_indices_and_shape_from_inames
    arg_base_indices, shape = find_var_base_indices_and_shape_from_inames(
            sfm_dom, [uarg+"'" for uarg in arg_names],
            kernel.cache_manager)
    print arg_names, shape

    # compute augmented domain

    # {{{ filter out unit-length dimensions

    non1_arg_names = []
    non1_arg_base_indices = []
    non1_shape = []

    for arg_name, bi, l in zip(arg_names, arg_base_indices, shape):
        if l > 1:
            non1_arg_names.append(arg_name)
            non1_arg_base_indices.append(bi)
            non1_shape.append(l)

    # }}}

    # {{{ subtract off the base indices
    # add the new, base-0 as new in dimensions

    sp = global_footprint_map.get_space()
    tgt_idx = sp.dim(dim_type.out)

    n_args = len(arg_names)
    nn1_args = len(non1_arg_names)

    aug_domain = global_footprint_map.move_dims(
            dim_type.out, tgt_idx,
            dim_type.in_, 0,
            n_args).range().coalesce()

    aug_domain = aug_domain.insert_dims(dim_type.set, tgt_idx, nn1_args)
    for i, name in enumerate(non1_arg_names):
        aug_domain = aug_domain.set_dim_name(dim_type.set, tgt_idx+i, name)

    # index layout now:
    # <....out.....> (tgt_idx) <base-0 non-1-length args> <args>

    from loopy.symbolic import aff_from_expr
    for arg_name, bi, s in zip(arg_names, arg_base_indices, shape):
        if s > 1:
            cns = isl.Constraint.equality_from_aff(
                    aff_from_expr(aug_domain.get_space(),
                        var(arg_name) - (var(arg_name+"'") - bi)))

            aug_domain = aug_domain.add_constraint(cns)

    # }}}

    # eliminate inames with non-zero base indices

    aug_domain = aug_domain.eliminate(dim_type.set, tgt_idx+nn1_args, n_args)
    aug_domain = aug_domain.remove_dims(dim_type.set, tgt_idx+nn1_args, n_args)

    base_indices_2, shape_2 = find_var_base_indices_and_shape_from_inames(
            aug_domain, non1_arg_names, kernel.cache_manager)

    assert base_indices_2 == [0] * nn1_args
    assert shape_2 == non1_shape

    return (non1_arg_names, aug_domain,
            arg_base_indices, non1_arg_base_indices, non1_shape)





def simplify_via_aff(space, expr):
    from loopy.symbolic import aff_from_expr, aff_to_expr
    return aff_to_expr(aff_from_expr(space, expr))




def precompute(kernel, subst_name, dtype, sweep_inames=[],
        new_arg_names=None, arg_name_to_tag={}, default_tag="l.auto"):

    subst = kernel.substitutions[subst_name]
    arg_names = subst.arguments
    subst_expr = subst.expression

    # {{{ gather up invocations

    invocation_descriptors = []
    invocation_arg_deps = set()

    def gather_substs(expr, name, args, rec):
        arg_deps = get_dependencies(args)
        if not arg_deps <= kernel.all_inames():
            raise RuntimeError("CSE arguments in '%s' do not consist "
                    "exclusively of inames" % expr)

        invocation_arg_deps.update(arg_deps)

        invocation_descriptors.append(
                InvocationDescriptor(expr=expr, args=args))
        return expr

    from loopy.symbolic import SubstitutionCallbackMapper
    scm = SubstitutionCallbackMapper([subst_name], gather_substs)
    for insn in kernel.instructions:
        scm(insn.expression)
    for s in kernel.substitutions.itervalues():
        if s is not subst:
            scm(s.expression)

    allowable_sweep_inames = invocation_arg_deps | set(arg_names)
    if not set(sweep_inames) <= allowable_sweep_inames:
        raise RuntimeError("independent iname(s) '%s' do not occur as arg names "
                "of subsitution rule or in arguments of invocation" % (",".join(
                    set(sweep_inames)-allowable_sweep_inames)))

    # }}}

    # {{{ process ind_iname_to_tag argument

    arg_name_to_tag = arg_name_to_tag.copy()

    from loopy.kernel import parse_tag
    default_tag = parse_tag(default_tag)
    for iname in arg_names:
        arg_name_to_tag.setdefault(iname, default_tag)

    if not set(arg_name_to_tag.iterkeys()) <= set(arg_names):
        raise RuntimeError("tags for non-argument names may not be passed")

    # here, all information is consolidated into ind_iname_to_tag

    # }}}

    newly_created_var_names = set()

    # {{{ make sure that new arg names are unique

    # (and substitute in subst_expressions if any variable name changes are necessary)

    old_to_new = {}

    unique_new_arg_names = []
    new_arg_name_to_tag = {}
    for i, name in enumerate(arg_names):
        new_name = None

        if new_arg_names is not None and i < len(new_arg_names):
            new_name = new_arg_names[i]
            if new_name in kernel.all_variable_names():
                raise RuntimeError("new name '%s' already exists" % new_name)

        if name in kernel.all_variable_names():
            based_on = "%s_%s" % (name, subst_name)
            new_name = kernel.make_unique_var_name(
                    based_on=based_on, extra_used_vars=newly_created_var_names)

        if new_name is not None:
            old_to_new[name] = var(new_name)
            unique_new_arg_names.append(new_name)
            new_arg_name_to_tag[new_name] = arg_name_to_tag[name]
            newly_created_var_names.add(new_name)
        else:
            unique_new_arg_names.append(name)
            new_arg_name_to_tag[name] = arg_name_to_tag[name]
            newly_created_var_names.add(name)

    old_arg_names = arg_names
    arg_names = unique_new_arg_names

    arg_name_to_tag = new_arg_name_to_tag
    subst_expr = (
            SubstitutionMapper(make_subst_func(old_to_new))
            (subst_expr))

    # }}}

    # {{{ align and intersect the footprint and the domain

    # (If there are independent inames, this adds extra dimensions to the domain.)
    (non1_arg_names, new_domain,
                arg_base_indices, non1_arg_base_indices, non1_shape) = \
                        get_footprint(kernel, subst_name, old_arg_names, arg_names,
                                sweep_inames, invocation_descriptors)

    new_domain = new_domain.coalesce()
    if isinstance(new_domain, isl.Set):
        dom_bsets = new_domain.get_basic_sets()
        if len(dom_bsets) > 1:
            raise NotImplementedError("Substitution '%s' yielded a non-convex footprint"
                    % subst_name)

        new_domain, = dom_bsets

    # }}}

    # {{{ set up temp variable

    target_var_name = kernel.make_unique_var_name(based_on=subst_name,
            extra_used_vars=newly_created_var_names)

    from loopy.kernel import TemporaryVariable

    new_temporary_variables = kernel.temporary_variables.copy()
    new_temporary_variables[target_var_name] = TemporaryVariable(
            name=target_var_name,
            dtype=np.dtype(dtype),
            base_indices=(0,)*len(non1_shape),
            shape=non1_shape,
            is_local=None)

    # }}}

    # {{{ set up compute insn

    assignee = var(target_var_name)

    if non1_arg_names:
        assignee = assignee[tuple(var(iname) for iname in non1_arg_names)]

    def zero_length_1_arg(arg_name):
        if arg_name in non1_arg_names:
            return var(arg_name)
        else:
            return 0

    compute_expr = (SubstitutionMapper(
        make_subst_func(dict(
            (arg_name, zero_length_1_arg(arg_name)+bi)
            for arg_name, bi in zip(arg_names, arg_base_indices)
            )))
        (subst_expr))

    from loopy.kernel import Instruction
    compute_insn = Instruction(
            id=kernel.make_unique_instruction_id(based_on=subst_name),
            assignee=assignee,
            expression=compute_expr)

    # }}}

    # {{{ substitute rule into instructions

    def do_substs(expr, name, args, rec):
        found = False
        for invdesc in invocation_descriptors:
            if expr is invdesc.expr:
                found = True
                break

        if not found:
            return

        args = [simplify_via_aff(new_domain.get_space(), arg-bi)
                for arg, bi in zip(args, non1_arg_base_indices)]

        new_outer_expr = var(target_var_name)
        if args:
            new_outer_expr = new_outer_expr[tuple(args)]

        return new_outer_expr
        # can't nest, don't recurse

    new_insns = [compute_insn]

    sub_map = SubstitutionCallbackMapper([subst_name], do_substs)
    for insn in kernel.instructions:
        new_insns.append(insn.copy(expression=sub_map(insn.expression)))

    # }}}

    new_iname_to_tag = kernel.iname_to_tag.copy()
    if sweep_inames:
        new_iname_to_tag.update(arg_name_to_tag)

    new_substs = dict(
            (s.name, s.copy(expression=sub_map(subst.expression)))
            for s in kernel.substitutions.itervalues())
    del new_substs[subst_name]

    return kernel.copy(
            domain=new_domain,
            instructions=new_insns,
            substitutions=new_substs,
            temporary_variables=new_temporary_variables,
            iname_to_tag=new_iname_to_tag)





# vim: foldmethod=marker

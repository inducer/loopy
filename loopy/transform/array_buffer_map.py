__copyright__ = "Copyright (C) 2012-2015 Andreas Kloeckner"

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


from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
from typing import Optional, Callable, Sequence, Tuple, Any
from typing_extensions import Self
import islpy as isl
from islpy import dim_type
from loopy.symbolic import (get_dependencies, SubstitutionMapper)
from pymbolic.mapper.substitutor import make_subst_func

from pytools import memoize_method
from pymbolic import var

from loopy.typing import ExpressionT


@dataclass(frozen=True)
class AccessDescriptor:
    """
    .. attribute:: identifier

        An identifier under user control, used to connect this access descriptor
        to the access that generated it. Any Python value.
    """

    identifier: Any = None
    storage_axis_exprs: Optional[Sequence[ExpressionT]] = None

    def copy(self, **kwargs) -> Self:
        return replace(self, **kwargs)


def to_parameters_or_project_out(param_inames, set_inames, set):
    for iname in list(set.get_space().get_var_dict().keys()):
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


# {{{ construct storage->sweep map

def build_per_access_storage_to_domain_map(
        storage_axis_exprs: Sequence[ExpressionT],
        domain: isl.BasicSet,
        storage_axis_names: Sequence[str],
        prime_sweep_inames: Callable[[ExpressionT], ExpressionT]
        ) -> isl.BasicMap:

    map_space = domain.space
    stor_dim = len(storage_axis_names)
    rn = map_space.dim(dim_type.out)

    map_space = map_space.add_dims(dim_type.in_, stor_dim)
    for i, saxis in enumerate(storage_axis_names):
        # arg names are initially primed, to be replaced with unprimed
        # base-0 versions below

        map_space = map_space.set_dim_name(dim_type.in_, i, saxis+"'")

    # map_space: [stor_axes'] -> [domain](dup_sweep_index)[dup_sweep](rn)

    set_space = map_space.move_dims(
            dim_type.out, rn,
            dim_type.in_, 0, stor_dim).range()

    # set_space: [domain](dup_sweep_index)[dup_sweep](rn)[stor_axes']

    stor2sweep = None

    from loopy.symbolic import guarded_aff_from_expr

    for saxis, sa_expr in zip(storage_axis_names, storage_axis_exprs):
        cns_expr = var(saxis+"'") - prime_sweep_inames(sa_expr)
        cns_aff = guarded_aff_from_expr(set_space, cns_expr)
        cns = isl.Constraint.equality_from_aff(cns_aff)

        cns_map = isl.BasicMap.from_constraint(cns)
        if stor2sweep is None:
            stor2sweep = cns_map
        else:
            stor2sweep = stor2sweep.intersect(cns_map)

    if stor2sweep is not None:
        stor2sweep = stor2sweep.move_dims(
                dim_type.in_, 0,
                dim_type.out, rn, stor_dim)

    # stor2sweep is back in map_space
    return stor2sweep


def move_to_par_from_out(s2smap, except_inames):
    while True:
        var_dict = s2smap.get_var_dict(dim_type.out)
        todo_inames = set(var_dict) - except_inames
        if todo_inames:
            iname = todo_inames.pop()

            _, dim_idx = var_dict[iname]
            s2smap = s2smap.move_dims(
                    dim_type.param, s2smap.dim(dim_type.param),
                    dim_type.out, dim_idx, 1)
        else:
            return s2smap


def build_global_storage_to_sweep_map(access_descriptors,
        domain_dup_sweep, storage_axis_names, prime_sweep_inames):
    # The storage map goes from storage axes to the domain.
    # The first len(arg_names) storage dimensions are the rule's arguments.

    global_stor2sweep = None

    # build footprint
    for accdesc in access_descriptors:
        stor2sweep = build_per_access_storage_to_domain_map(
                accdesc.storage_axis_exprs, domain_dup_sweep,
                storage_axis_names,
                prime_sweep_inames)

        if global_stor2sweep is None:
            global_stor2sweep = stor2sweep
        else:
            global_stor2sweep = global_stor2sweep.union(stor2sweep)

    if isinstance(global_stor2sweep, isl.BasicMap):
        global_stor2sweep = isl.Map.from_basic_map(global_stor2sweep)
    global_stor2sweep = global_stor2sweep.intersect_range(domain_dup_sweep)

    # space for global_stor2sweep:
    # [stor_axes'] -> [domain](dup_sweep_index)[dup_sweep](rn)

    return global_stor2sweep

# }}}


# {{{ compute storage bounds

def find_var_base_indices_and_shape_from_inames(
        domain, inames, cache_manager, context=None,
        n_allowed_params_in_shape=None):
    base_indices_and_sizes = [
            cache_manager.base_index_and_length(
                domain, iname, context,
                n_allowed_params_in_length=n_allowed_params_in_shape)
            for iname in inames]
    return list(zip(*base_indices_and_sizes))


def compute_bounds(kernel, domain, stor2sweep,
        primed_sweep_inames, storage_axis_names):

    bounds_footprint_map = move_to_par_from_out(
            stor2sweep, except_inames=frozenset(primed_sweep_inames))

    # compute bounds for each storage axis
    storage_domain = bounds_footprint_map.domain().coalesce()

    if not storage_domain.is_bounded():
        raise RuntimeError("sweep did not result in a bounded storage domain")

    return find_var_base_indices_and_shape_from_inames(
            storage_domain, [saxis+"'" for saxis in storage_axis_names],
            kernel.cache_manager, context=kernel.assumptions,
            n_allowed_params_in_shape=stor2sweep.dim(isl.dim_type.param))

# }}}


# {{{ array-to-buffer map

class ArrayToBufferMapBase(ABC):
    non1_storage_axis_names: Tuple[str, ...]
    storage_base_indices: Tuple[ExpressionT, ...]
    non1_storage_shape: Tuple[ExpressionT, ...]
    non1_storage_axis_flags: Tuple[ExpressionT, ...]

    @abstractmethod
    def is_access_descriptor_in_footprint(self, accdesc: AccessDescriptor) -> bool:
        ...

    @abstractmethod
    def augment_domain_with_sweep(self, domain, new_non1_storage_axis_names,
            boxify_sweep=False):
        ...


class ArrayToBufferMap(ArrayToBufferMapBase):
    def __init__(self, kernel, domain, sweep_inames, access_descriptors,
            storage_axis_count):
        self.kernel = kernel
        self.sweep_inames = sweep_inames

        storage_axis_names = self.storage_axis_names = [
                "_loopy_storage_%d" % i for i in range(storage_axis_count)]

        # {{{ duplicate sweep inames

        # The duplication is necessary, otherwise the storage fetch
        # inames remain weirdly tied to the original sweep inames.

        self.primed_sweep_inames = [psin+"'" for psin in sweep_inames]

        from loopy.isl_helpers import duplicate_axes
        dup_sweep_index = domain.space.dim(dim_type.out)
        domain_dup_sweep = duplicate_axes(
                domain, sweep_inames,
                self.primed_sweep_inames)

        self.prime_sweep_inames = SubstitutionMapper(make_subst_func(
            {sin: var(psin)
                for sin, psin in zip(sweep_inames, self.primed_sweep_inames)}))

        # # }}}

        self.stor2sweep = build_global_storage_to_sweep_map(
                access_descriptors,
                domain_dup_sweep,
                storage_axis_names,
                self.prime_sweep_inames)

        storage_base_indices, storage_shape = compute_bounds(
                kernel, domain, self.stor2sweep, self.primed_sweep_inames,
                storage_axis_names)

        # compute augmented domain

        # {{{ filter out unit-length dimensions

        non1_storage_axis_flags = []
        non1_storage_shape = []

        for saxis_len in storage_shape:
            has_length_non1 = saxis_len != 1

            non1_storage_axis_flags.append(has_length_non1)

            if has_length_non1:
                non1_storage_shape.append(saxis_len)

        # }}}

        # {{{ subtract off the base indices
        # add the new, base-0 indices as new in dimensions

        sp = self.stor2sweep.get_space()
        stor_idx = sp.dim(dim_type.out)

        n_stor = storage_axis_count
        nn1_stor = len(non1_storage_shape)

        aug_domain = self.stor2sweep.move_dims(
                dim_type.out, stor_idx,
                dim_type.in_, 0,
                n_stor).range()

        # aug_domain space now:
        # [domain](dup_sweep_index)[dup_sweep](stor_idx)[stor_axes']

        aug_domain = aug_domain.insert_dims(dim_type.set, stor_idx, nn1_stor)

        inew = 0
        for i, name in enumerate(storage_axis_names):
            if non1_storage_axis_flags[i]:
                aug_domain = aug_domain.set_dim_name(
                        dim_type.set, stor_idx + inew, name)
                inew += 1

        # aug_domain space now:
        # [domain](dup_sweep_index)[dup_sweep](stor_idx)[stor_axes'][n1_stor_axes]

        from loopy.symbolic import aff_from_expr
        for saxis, bi, s in zip(storage_axis_names, storage_base_indices,
                storage_shape):
            if s != 1:
                cns = isl.Constraint.equality_from_aff(
                        aff_from_expr(aug_domain.get_space(),
                            var(saxis) - (var(saxis+"'") - bi)))

                aug_domain = aug_domain.add_constraint(cns)

        # }}}

        # eliminate (primed) storage axes with non-zero base indices
        aug_domain = aug_domain.project_out(dim_type.set, stor_idx+nn1_stor, n_stor)

        # eliminate duplicated sweep_inames
        nsweep = len(sweep_inames)
        aug_domain = aug_domain.project_out(dim_type.set, dup_sweep_index, nsweep)

        self.non1_storage_axis_flags = non1_storage_axis_flags
        self.aug_domain = aug_domain
        self.storage_base_indices = storage_base_indices
        self.non1_storage_shape = tuple(non1_storage_shape)

    def augment_domain_with_sweep(self, domain, new_non1_storage_axis_names,
            boxify_sweep=False):

        renamed_aug_domain = self.aug_domain
        first_storage_index = (renamed_aug_domain.dim(dim_type.set)
                - len(self.non1_storage_shape))

        inon1 = 0
        for i, old_name in enumerate(self.storage_axis_names):
            if not self.non1_storage_axis_flags[i]:
                continue

            new_name = new_non1_storage_axis_names[inon1]

            assert (
                    renamed_aug_domain.get_dim_name(
                        dim_type.set, first_storage_index+inon1)
                    == old_name)
            renamed_aug_domain = renamed_aug_domain.set_dim_name(
                    dim_type.set, first_storage_index+inon1, new_name)

            inon1 += 1

        # Order of arguments to align_two matters--'domain' should be the
        # 'guiding' ordering.
        renamed_aug_domain, domain = isl.align_two(renamed_aug_domain, domain)

        domain = domain & renamed_aug_domain

        from loopy.isl_helpers import convexify, boxify
        if boxify_sweep:
            return boxify(self.kernel.cache_manager, domain,
                    new_non1_storage_axis_names, self.kernel.assumptions)
        else:
            return convexify(domain)

    def is_access_descriptor_in_footprint(self, accdesc: AccessDescriptor) -> bool:
        assert accdesc.storage_axis_exprs is not None
        return self._is_access_descriptor_in_footprint_inner(
                tuple(accdesc.storage_axis_exprs))

    @memoize_method
    def _is_access_descriptor_in_footprint_inner(self, storage_axis_exprs):
        # Make all inames except the sweep parameters. (The footprint may depend on
        # those.) (I.e. only leave sweep inames as out parameters.)

        global_s2s_par_dom = move_to_par_from_out(
                self.stor2sweep,
                except_inames=frozenset(self.primed_sweep_inames)).domain()

        arg_inames = set(global_s2s_par_dom.get_var_names(dim_type.param))

        for arg in storage_axis_exprs:
            arg_inames.update(get_dependencies(arg))
        arg_inames = frozenset(arg_inames & self.kernel.all_inames())

        from loopy.kernel import CannotBranchDomainTree
        try:
            usage_domain = self.kernel.get_inames_domain(arg_inames)
        except CannotBranchDomainTree:
            return False

        for i in range(usage_domain.dim(dim_type.set)):
            iname = usage_domain.get_dim_name(dim_type.set, i)
            if iname in self.sweep_inames:
                usage_domain = usage_domain.set_dim_name(
                        dim_type.set, i, iname+"'")

        stor2sweep = build_per_access_storage_to_domain_map(
                storage_axis_exprs,
                usage_domain, self.storage_axis_names,
                self.prime_sweep_inames)

        if stor2sweep is None:
            # happens if there are no indices
            # -> yes, in footprint
            return True

        if isinstance(stor2sweep, isl.BasicMap):
            stor2sweep = isl.Map.from_basic_map(stor2sweep)

        stor2sweep = stor2sweep.intersect_range(usage_domain)

        stor2sweep = move_to_par_from_out(stor2sweep,
                except_inames=frozenset(self.primed_sweep_inames))

        s2s_domain = stor2sweep.domain()
        s2s_domain, aligned_g_s2s_parm_dom = isl.align_two(
                s2s_domain, global_s2s_par_dom)

        arg_restrictions = (
                aligned_g_s2s_parm_dom
                .eliminate(dim_type.set, 0,
                    aligned_g_s2s_parm_dom.dim(dim_type.set))
                .remove_divs())

        return (arg_restrictions & s2s_domain).is_subset(
                aligned_g_s2s_parm_dom)


class NoOpArrayToBufferMap(ArrayToBufferMapBase):
    non1_storage_axis_names = ()
    storage_base_indices = ()
    non1_storage_shape = ()

    def is_access_descriptor_in_footprint(self, accdesc: AccessDescriptor) -> bool:
        # no index dependencies--every reference to the subst rule
        # is necessarily in the footprint.

        return True

    def augment_domain_with_sweep(self, domain, new_non1_storage_axis_names,
            boxify_sweep=False):
        return domain
# }}}

# vim: foldmethod=marker

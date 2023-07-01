__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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


from loopy.diagnostic import LoopyError
from loopy.translation_unit import for_each_kernel
import pymbolic

import logging
logger = logging.getLogger(__name__)


__doc__ = """

.. currentmodule:: loopy

.. autofunction:: privatize_temporaries_with_inames
.. autofunction:: unprivatize_temporaries_with_inames
"""


# {{{ privatize temporaries with iname

from loopy.symbolic import IdentityMapper


class ExtraInameIndexInserter(IdentityMapper):
    def __init__(self, var_to_new_inames, iname_to_lbound):
        self.var_to_new_inames = var_to_new_inames
        self.iname_to_lbound = iname_to_lbound
        self.seen_priv_axis_inames = set()
        super().__init__()

    def map_subscript(self, expr):
        try:
            extra_idx = self.var_to_new_inames[expr.aggregate.name]
        except KeyError:
            return IdentityMapper.map_subscript(self, expr)
        else:
            index = expr.index
            if not isinstance(index, tuple):
                index = (index,)
            index = tuple(self.rec(i) for i in index)

            self.seen_priv_axis_inames.update(v.name for v in extra_idx)

            new_idx = index + tuple(v - self.iname_to_lbound[v.name]
                            for v in extra_idx)

            if len(new_idx) == 1:
                new_idx = new_idx[0]
            return expr.aggregate.index(new_idx)

    def map_variable(self, expr):
        try:
            new_idx = self.var_to_new_inames[expr.name]
        except KeyError:
            return expr
        else:
            self.seen_priv_axis_inames.update(v.name for v in new_idx)

            new_idx = tuple(v - self.iname_to_lbound[v.name]
                            for v in new_idx)

            if len(new_idx) == 1:
                new_idx = new_idx[0]
            return expr.index(new_idx)


@for_each_kernel
def privatize_temporaries_with_inames(
        kernel, privatizing_inames, only_var_names=None):
    """This function provides each loop iteration of the *privatizing_inames*
    with its own private entry in the temporaries it accesses (possibly
    restricted to *only_var_names*).

    This is accomplished implicitly as part of generating instruction-level
    parallelism by the "ILP" tag and accessible separately through this
    transformation.

    Example::

        for imatrix, i
            acc = 0
            for k
                acc = acc + a[imatrix, i, k] * vec[k]
            end
        end

    might become::

        for imatrix, i
            acc[imatrix] = 0
            for k
                acc[imatrix] = acc[imatrix] + a[imatrix, i, k] * vec[k]
            end
        end

    facilitating loop interchange of the *imatrix* loop.
    .. versionadded:: 2018.1
    """

    if isinstance(privatizing_inames, str):
        privatizing_inames = frozenset(
                s.strip()
                for s in privatizing_inames.split(","))

    if isinstance(only_var_names, str):
        only_var_names = frozenset(
                s.strip()
                for s in only_var_names.split(","))

    # {{{ sanity checks

    if (only_var_names is not None
            and privatizing_inames <= kernel.all_inames()
            and not (frozenset(only_var_names) <= kernel.all_variable_names())):
        raise LoopyError(f"Some variables in '{only_var_names}'"
                         f" not used in kernel '{kernel.name}'.")

    # }}}

    wmap = kernel.writer_map()

    var_to_new_priv_axis_iname = {}

    # {{{ find variables that need extra indices

    for tv in kernel.temporary_variables.values():
        if only_var_names is not None and tv.name not in only_var_names:
            continue

        for writer_insn_id in wmap.get(tv.name, []):
            writer_insn = kernel.id_to_insn[writer_insn_id]

            priv_axis_inames = writer_insn.within_inames & privatizing_inames

            referenced_priv_axis_inames = (priv_axis_inames
                    & writer_insn.write_dependency_names())

            new_priv_axis_inames = priv_axis_inames - referenced_priv_axis_inames

            if not new_priv_axis_inames:
                break

            if tv.name in var_to_new_priv_axis_iname:
                if new_priv_axis_inames != set(var_to_new_priv_axis_iname[tv.name]):
                    raise LoopyError("instruction '%s' requires adding "
                            "indices for privatizing var '%s' on iname(s) '%s', "
                            "but previous instructions required different "
                            "inames '%s'"
                            % (writer_insn_id, tv.name,
                                ", ".join(new_priv_axis_inames),
                                ", ".join(var_to_new_priv_axis_iname[tv.name])))

                continue

            var_to_new_priv_axis_iname[tv.name] = set(new_priv_axis_inames)

    # }}}

    # {{{ find ilp iname lengths

    from loopy.isl_helpers import static_max_of_pw_aff
    from loopy.symbolic import pw_aff_to_expr

    priv_axis_iname_to_length = {}
    iname_to_lbound = {}
    for priv_axis_inames in var_to_new_priv_axis_iname.values():
        for iname in priv_axis_inames:
            if iname in priv_axis_iname_to_length:
                continue

            bounds = kernel.get_iname_bounds(iname, constants_only=False)
            priv_axis_iname_to_length[iname] = pw_aff_to_expr(
                        static_max_of_pw_aff(bounds.size, constants_only=False))
            iname_to_lbound[iname] = pw_aff_to_expr(bounds.lower_bound_pw_aff)

    # }}}

    # {{{ change temporary variables

    from loopy.kernel.data import VectorizeTag

    new_temp_vars = kernel.temporary_variables.copy()
    for tv_name, inames in var_to_new_priv_axis_iname.items():
        tv = new_temp_vars[tv_name]
        extra_shape = tuple(priv_axis_iname_to_length[iname] for iname in inames)

        shape = tv.shape
        if shape is None:
            shape = ()

        dim_tags = ["c"] * (len(shape) + len(extra_shape))
        for i, iname in enumerate(inames):
            if kernel.iname_tags_of_type(iname, VectorizeTag):
                dim_tags[len(shape) + i] = "vec"

        base_indices = tv.base_indices
        if base_indices is not None:
            base_indices = base_indices + tuple([0]*len(extra_shape))

        new_temp_vars[tv.name] = tv.copy(shape=shape + extra_shape,
                base_indices=base_indices,
                # Forget what you knew about data layout,
                # create from scratch.
                dim_tags=dim_tags,
                dim_names=None)

    # }}}

    from pymbolic import var
    var_to_extra_iname = {
            var_name: tuple(var(iname) for iname in inames)
            for var_name, inames in var_to_new_priv_axis_iname.items()}

    new_insns = []

    for insn in kernel.instructions:
        eiii = ExtraInameIndexInserter(var_to_extra_iname,
                                       iname_to_lbound)
        new_insn = insn.with_transformed_expressions(eiii)
        if not eiii.seen_priv_axis_inames <= insn.within_inames:
            raise LoopyError(
                    "Kernel '%s': Instruction '%s': touched variable that "
                    "(for privatization, e.g. as performed for ILP) "
                    "required iname(s) '%s', but that the instruction was not "
                    "previously within the iname(s). To remedy this, first promote"
                    "the instruction into the iname."
                    % (kernel.name, insn.id, ", ".join(
                        eiii.seen_priv_axis_inames - insn.within_inames)))

        new_insns.append(new_insn)

    return kernel.copy(
        temporary_variables=new_temp_vars,
        instructions=new_insns)

# }}}


# {{{ unprivatize temporaries with iname

class _InameRemover(IdentityMapper):
    def __init__(self, inames_to_remove, only_var_names):
        self.only_var_names = only_var_names
        self.inames_to_remove = inames_to_remove
        self.var_name_to_remove_indices = {}
        super().__init__()

    def map_subscript(self, expr, in_subscript=False):
        name = expr.aggregate.name
        if not self.only_var_names or name in self.only_var_names:
            index = expr.index
            if not isinstance(index, tuple):
                index = (index,)

            remove_indices = {}
            new_index = []
            for i, index_expr in enumerate(index):
                if isinstance(index_expr, pymbolic.primitives.Variable) and \
                        index_expr.name in self.inames_to_remove:
                    remove_indices[i] = index_expr.name
                else:
                    new_index.append(index_expr)

            if name in self.var_name_to_remove_indices:
                old_remove_indices = self.var_name_to_remove_indices[name]
                if old_remove_indices != remove_indices:
                    raise LoopyError(f"Cannot remove indices {remove_indices}"
                         f" for subscript {expr} because there was another"
                         f" subscript that required removing different indices"
                         f" {old_remove_indices}")
            else:
                self.var_name_to_remove_indices[name] = remove_indices

            if new_index:
                if len(new_index) == 1:
                    new_index = new_index[0]
                else:
                    new_index = tuple(new_index)
                return expr.aggregate.index(new_index)
            else:
                return expr.aggregate
        else:
            return IdentityMapper.map_subscript(self, expr, in_subscript=False)


@for_each_kernel
def unprivatize_temporaries_with_inames(
        kernel, privatizing_inames, only_var_names=None):
    """This function reverses the effects of
    :func:`privatize_temporaries_with_inames` and removes the private entries
    in the temporaries each loop iteration of the *privatizing_inames*
    accesses (possibly restricted to *only_var_names*).

    Example::

        for imatrix, i
            acc[imatrix] = 0
            for k
                acc[imatrix] = acc[imatrix] + a[imatrix, i, k] * vec[k]
            end
        end

    might become::

        for imatrix, i
            acc = 0
            for k
                acc = acc + a[imatrix, i, k] * vec[k]
            end
        end

    .. versionadded:: 2022.1
    """

    if isinstance(privatizing_inames, str):
        privatizing_inames = frozenset(
                s.strip()
                for s in privatizing_inames.split(","))

    if isinstance(only_var_names, str):
        only_var_names = frozenset(
                s.strip()
                for s in only_var_names.split(","))

    # {{{ sanity checks

    if (only_var_names is not None
            and privatizing_inames <= kernel.all_inames()
            and not (frozenset(only_var_names) <= kernel.all_variable_names())):
        raise LoopyError(f"Some variables in '{only_var_names}'"
                         f" not used in kernel '{kernel.name}'.")

    # }}}

    ir = _InameRemover(privatizing_inames, only_var_names)

    new_insns = [
        insn.with_transformed_expressions(lambda x: ir(x, False))
        for insn in kernel.instructions]

    # {{{ change temporary variables

    var_name_to_remove_indices = ir.var_name_to_remove_indices

    from loopy.kernel.array import VectorArrayDimTag

    new_temp_vars = kernel.temporary_variables.copy()
    for tv_name, tv in new_temp_vars.items():
        remove_indices = var_name_to_remove_indices.get(tv_name, {})
        new_shape = tv.shape
        if new_shape is not None:
            new_shape = tuple(dim for idim, dim in enumerate(new_shape)
                if idim not in remove_indices)

        new_dim_tags = tv.dim_tags
        if new_dim_tags is not None:
            new_dim_tags = ["vec" if isinstance(dim_tag, VectorArrayDimTag) else "c"
                            for idim, dim_tag in enumerate(new_dim_tags)]
            new_dim_tags = tuple(dim for idim, dim in enumerate(new_dim_tags)
                if idim not in remove_indices)

        new_dim_names = tv.dim_names
        if new_dim_names is not None:
            new_dim_names = tuple(dim for idim, dim in enumerate(new_dim_names)
                if idim not in remove_indices)

        new_base_indices = tv.base_indices
        if new_base_indices is not None:
            new_base_indices = tuple(dim for idim, dim in enumerate(new_base_indices)
                if idim not in remove_indices)

        new_temp_vars[tv_name] = tv.copy(
                shape=new_shape,
                dim_tags=new_dim_tags,
                dim_names=new_dim_names,
                base_indices=new_base_indices)

    # }}}

    return kernel.copy(
        temporary_variables=new_temp_vars,
        instructions=new_insns)

# }}}


# vim: foldmethod=marker

from __future__ import division, absolute_import

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


import six
from loopy.diagnostic import LoopyError

import logging
logger = logging.getLogger(__name__)


__doc__ = """

.. currentmodule:: loopy

.. autofunction:: privatize_temporaries_with_inames
"""


# {{{ privatize temporaries with iname

from loopy.symbolic import IdentityMapper


class ExtraInameIndexInserter(IdentityMapper):
    def __init__(self, var_to_new_inames):
        self.var_to_new_inames = var_to_new_inames
        self.seen_priv_axis_inames = set()

    def map_subscript(self, expr):
        try:
            new_idx = self.var_to_new_inames[expr.aggregate.name]
        except KeyError:
            return IdentityMapper.map_subscript(self, expr)
        else:
            index = expr.index
            if not isinstance(index, tuple):
                index = (index,)
            index = tuple(self.rec(i) for i in index)

            self.seen_priv_axis_inames.update(v.name for v in new_idx)
            return expr.aggregate.index(index + new_idx)

    def map_variable(self, expr):
        try:
            new_idx = self.var_to_new_inames[expr.name]
        except KeyError:
            return expr
        else:
            self.seen_priv_axis_inames.update(v.name for v in new_idx)
            return expr.index(new_idx)


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

    wmap = kernel.writer_map()

    var_to_new_priv_axis_iname = {}

    # {{{ find variables that need extra indices

    for tv in six.itervalues(kernel.temporary_variables):
        if only_var_names is not None and tv.name not in only_var_names:
            continue

        for writer_insn_id in wmap.get(tv.name, []):
            writer_insn = kernel.id_to_insn[writer_insn_id]

            priv_axis_inames = kernel.insn_inames(writer_insn) & privatizing_inames

            referenced_priv_axis_inames = (priv_axis_inames
                    & writer_insn.write_dependency_names())

            new_priv_axis_inames = priv_axis_inames - referenced_priv_axis_inames

            if not new_priv_axis_inames:
                break

            if tv.name in var_to_new_priv_axis_iname:
                if new_priv_axis_inames != set(var_to_new_priv_axis_iname[tv.name]):
                    raise LoopyError("instruction '%s' requires adding "
                            "indices for privatizing var '%s' on iname(s) '%s', "
                            "but previous instructions required inames '%s'"
                            % (writer_insn_id, tv.name,
                                ", ".join(new_priv_axis_inames),
                                ", ".join(var_to_new_priv_axis_iname[tv.name])))

                continue

            var_to_new_priv_axis_iname[tv.name] = set(new_priv_axis_inames)

    # }}}

    # {{{ find ilp iname lengths

    from loopy.isl_helpers import static_max_of_pw_aff, static_min_of_pw_aff
    from loopy.symbolic import pw_aff_to_expr

    priv_axis_iname_to_length = {}
    for priv_axis_inames in six.itervalues(var_to_new_priv_axis_iname):
        for iname in priv_axis_inames:
            if iname in priv_axis_iname_to_length:
                continue
            bounds = kernel.get_iname_bounds(iname, constants_only=False)
            priv_axis_iname_to_length[iname] = pw_aff_to_expr(
                        static_max_of_pw_aff(bounds.size, constants_only=False))
            assert (static_max_of_pw_aff(
                bounds.lower_bound_pw_aff, constants_only=False).plain_is_zero() or
                static_min_of_pw_aff(
                    bounds.lower_bound_pw_aff, constants_only=False).plain_is_zero())

    # }}}

    # {{{ change temporary variables

    from loopy.kernel.data import VectorizeTag, CVectorizeTag

    new_temp_vars = kernel.temporary_variables.copy()
    for tv_name, inames in six.iteritems(var_to_new_priv_axis_iname):
        tv = new_temp_vars[tv_name]
        extra_shape = tuple(priv_axis_iname_to_length[iname] for iname in inames)

        shape = tv.shape
        if shape is None:
            shape = ()

        dim_tags = ["c"] * (len(shape) + len(extra_shape))
        for i, iname in enumerate(inames):
            if kernel.iname_tags_of_type(iname, VectorizeTag):
                dim_tags[len(shape) + i] = "vec"
            elif kernel.iname_tags_of_type(iname, CVectorizeTag):
                dim_tags[len(shape) + i] = "c_vec"

        new_temp_vars[tv.name] = tv.copy(shape=shape + extra_shape,
                # Forget what you knew about data layout,
                # create from scratch.
                dim_tags=dim_tags,
                dim_names=None)

    # }}}

    from pymbolic import var
    var_to_extra_iname = dict(
            (var_name, tuple(var(iname) for iname in inames))
            for var_name, inames in six.iteritems(var_to_new_priv_axis_iname))

    new_insns = []

    for insn in kernel.instructions:
        eiii = ExtraInameIndexInserter(var_to_extra_iname)
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


# vim: foldmethod=marker

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


# {{{ duplicate private vars for ilp and vec

from loopy.symbolic import IdentityMapper


class ExtraInameIndexInserter(IdentityMapper):
    def __init__(self, var_to_new_inames):
        self.var_to_new_inames = var_to_new_inames

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

            return expr.aggregate.index(index + new_idx)

    def map_variable(self, expr):
        try:
            new_idx = self.var_to_new_inames[expr.name]
        except KeyError:
            return expr
        else:
            return expr.index(new_idx)


def add_axes_to_temporaries_for_ilp_and_vec(kernel, iname=None):
    if iname is not None:
        logger.debug("%s: add axes to temporaries for ilp" % kernel.name)

    wmap = kernel.writer_map()

    from loopy.kernel.data import IlpBaseTag, VectorizeTag

    var_to_new_ilp_inames = {}

    # {{{ find variables that need extra indices

    for tv in six.itervalues(kernel.temporary_variables):
        for writer_insn_id in wmap.get(tv.name, []):
            writer_insn = kernel.id_to_insn[writer_insn_id]

            if iname is None:
                ilp_inames = frozenset(iname
                        for iname in kernel.insn_inames(writer_insn)
                        if isinstance(
                            kernel.iname_to_tag.get(iname),
                            (IlpBaseTag, VectorizeTag)))
            else:
                if not isinstance(
                        kernel.iname_to_tag.get(iname),
                        (IlpBaseTag, VectorizeTag)):
                    raise LoopyError(
                            "'%s' is not an ILP iname"
                            % iname)

                ilp_inames = frozenset([iname])

            referenced_ilp_inames = (ilp_inames
                    & writer_insn.write_dependency_names())

            new_ilp_inames = ilp_inames - referenced_ilp_inames

            if not new_ilp_inames:
                break

            if tv.name in var_to_new_ilp_inames:
                if new_ilp_inames != set(var_to_new_ilp_inames[tv.name]):
                    raise LoopyError("instruction '%s' requires adding "
                            "indices for ILP inames '%s' on var '%s', but previous "
                            "instructions required inames '%s'"
                            % (writer_insn_id, ", ".join(new_ilp_inames),
                                ", ".join(var_to_new_ilp_inames[tv.name])))

                continue

            var_to_new_ilp_inames[tv.name] = set(new_ilp_inames)

    # }}}

    # {{{ find ilp iname lengths

    from loopy.isl_helpers import static_max_of_pw_aff
    from loopy.symbolic import pw_aff_to_expr

    ilp_iname_to_length = {}
    for ilp_inames in six.itervalues(var_to_new_ilp_inames):
        for iname in ilp_inames:
            if iname in ilp_iname_to_length:
                continue

            bounds = kernel.get_iname_bounds(iname, constants_only=True)
            ilp_iname_to_length[iname] = int(pw_aff_to_expr(
                        static_max_of_pw_aff(bounds.size, constants_only=True)))

            assert static_max_of_pw_aff(
                    bounds.lower_bound_pw_aff, constants_only=True).plain_is_zero()

    # }}}

    # {{{ change temporary variables

    new_temp_vars = kernel.temporary_variables.copy()
    for tv_name, inames in six.iteritems(var_to_new_ilp_inames):
        tv = new_temp_vars[tv_name]
        extra_shape = tuple(ilp_iname_to_length[iname] for iname in inames)

        shape = tv.shape
        if shape is None:
            shape = ()

        dim_tags = ["c"] * (len(shape) + len(extra_shape))
        for i, iname in enumerate(inames):
            if isinstance(kernel.iname_to_tag.get(iname), VectorizeTag):
                dim_tags[len(shape) + i] = "vec"

        new_temp_vars[tv.name] = tv.copy(shape=shape + extra_shape,
                # Forget what you knew about data layout,
                # create from scratch.
                dim_tags=dim_tags,
                dim_names=None)

    # }}}

    from pymbolic import var
    eiii = ExtraInameIndexInserter(
            dict((var_name, tuple(var(iname) for iname in inames))
                for var_name, inames in six.iteritems(var_to_new_ilp_inames)))

    new_insns = [
            insn.with_transformed_expressions(eiii)
            for insn in kernel.instructions]

    return kernel.copy(
        temporary_variables=new_temp_vars,
        instructions=new_insns)

# }}}


# {{{ realize_ilp

def realize_ilp(kernel, iname):
    """Instruction-level parallelism (as realized by the loopy iname
    tag ``"ilp"``) provides the illusion that multiple concurrent
    program instances execute in lockstep within a single instruction
    stream.

    To do so, storage that is private to each instruction stream needs to be
    duplicated so that each program instance receives its own copy.  Storage
    that is written to in an instruction using an ILP iname but whose left-hand
    side indices do not contain said ILP iname is marked for duplication.

    This storage duplication is carried out automatically at code generation
    time, but, using this function, can also be carried out ahead of time
    on a per-iname basis (so that, for instance, data layout of the duplicated
    storage can be controlled explicitly.
    """
    from loopy.ilp import add_axes_to_temporaries_for_ilp_and_vec
    return add_axes_to_temporaries_for_ilp_and_vec(kernel, iname)

# }}}


# vim: foldmethod=marker

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


import islpy as isl


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
            upper_bset = isl.BasicSet.universe(space).add_constraint(
                isl.Constraint.inequality_from_aff(
                    iname_rel_aff(space,
                        iname, ">", upper_bound_aff-upper_incr)))
            if lower_incr:
                # Ensure that this slab is actually distinct from the
                # lower one, if it exists.
                _, lower_bset = lower_slab
                upper_bset, = upper_bset.subtract(lower_bset).get_basic_sets()
            upper_slab = ("final", upper_bset)
            upper_bulk_bound = (
                    isl.Constraint.inequality_from_aff(
                        iname_rel_aff(space,
                            iname, "<=", upper_bound_aff-upper_incr)))
        else:
            upper_slab = None

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


# vim: foldmethod=marker

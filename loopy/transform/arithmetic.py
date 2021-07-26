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


from loopy.diagnostic import LoopyError

from loopy.translation_unit import for_each_kernel
from loopy.kernel import LoopKernel


# {{{ fold constants

@for_each_kernel
def fold_constants(kernel):
    from loopy.symbolic import ConstantFoldingMapper
    cfm = ConstantFoldingMapper()

    new_insns = [
            insn.with_transformed_expressions(cfm)
            for insn in kernel.instructions]

    new_substs = {
            sub.name:
            sub.copy(expression=cfm(sub.expression))
            for sub in kernel.substitutions.values()}

    return kernel.copy(
            instructions=new_insns,
            substitutions=new_substs)

# }}}


# {{{ collect_common_factors_on_increment

# thus far undocumented
@for_each_kernel
def collect_common_factors_on_increment(kernel, var_name, vary_by_axes=()):
    assert isinstance(kernel, LoopKernel)
    # FIXME: Does not understand subst rules for now
    if kernel.substitutions:
        from loopy.transform.subst import expand_subst
        kernel = expand_subst(kernel)

    if var_name in kernel.temporary_variables:
        var_descr = kernel.temporary_variables[var_name]
    elif var_name in kernel.arg_dict:
        var_descr = kernel.arg_dict[var_name]
    else:
        raise NameError("array '%s' was not found" % var_name)

    # {{{ check/normalize vary_by_axes

    if isinstance(vary_by_axes, str):
        vary_by_axes = vary_by_axes.split(",")

    from loopy.kernel.array import ArrayBase
    if isinstance(var_descr, ArrayBase):
        if var_descr.dim_names is not None:
            name_to_index = {
                    name: idx
                    for idx, name in enumerate(var_descr.dim_names)}
        else:
            name_to_index = {}

        def map_ax_name_to_index(ax):
            if isinstance(ax, str):
                try:
                    return name_to_index[ax]
                except KeyError:
                    raise LoopyError("axis name '%s' not understood " % ax)
            else:
                return ax

        vary_by_axes = [map_ax_name_to_index(ax) for ax in vary_by_axes]

        if (
                vary_by_axes
                and
                (min(vary_by_axes) < 0
                or
                max(vary_by_axes) > var_descr.num_user_axes())):
            raise LoopyError("vary_by_axes refers to out-of-bounds axis index")

    # }}}

    from pymbolic.mapper.substitutor import make_subst_func
    from pymbolic.primitives import (Sum, Product, is_zero,
            flattened_sum, flattened_product, Subscript, Variable)
    from loopy.symbolic import (get_dependencies, SubstitutionMapper,
            UnidirectionalUnifier)

    # {{{ common factor key list maintenance

    # list of (index_key, common factors found)
    common_factors = []

    def find_unifiable_cf_index(index_key):
        for i, (key, _val) in enumerate(common_factors):
            unif = UnidirectionalUnifier(
                    lhs_mapping_candidates=get_dependencies(key))

            unif_result = unif(key, index_key)

            if unif_result:
                assert len(unif_result) == 1
                return i, unif_result[0]

        return None, None

    def extract_index_key(access_expr):
        if isinstance(access_expr, Variable):
            return ()

        elif isinstance(access_expr, Subscript):
            index = access_expr.index_tuple
            return tuple(index[ax] for ax in vary_by_axes)
        else:
            raise ValueError("unexpected type of access_expr")

    def is_assignee(insn):
        return var_name in insn.assignee_var_names()

    def iterate_as(cls, expr):
        if isinstance(expr, cls):
            yield from expr.children
        else:
            yield expr

    # }}}

    # {{{ find common factors

    from loopy.kernel.data import Assignment

    for insn in kernel.instructions:
        if not is_assignee(insn):
            continue

        if not isinstance(insn, Assignment):
            raise LoopyError("'%s' modified by non-single-assignment"
                    % var_name)

        lhs = insn.assignee
        rhs = insn.expression

        if is_zero(rhs):
            continue

        index_key = extract_index_key(lhs)
        cf_index, unif_result = find_unifiable_cf_index(index_key)

        if cf_index is None:
            # {{{ doesn't exist yet

            assert unif_result is None

            my_common_factors = None

            for term in iterate_as(Sum, rhs):
                if term == lhs:
                    continue

                for part in iterate_as(Product, term):
                    if var_name in get_dependencies(part):
                        raise LoopyError("unexpected dependency on '%s' "
                                "in RHS of instruction '%s'"
                                % (var_name, insn.id))

                product_parts = set(iterate_as(Product, term))

                if my_common_factors is None:
                    my_common_factors = product_parts
                else:
                    my_common_factors = my_common_factors & product_parts

            if my_common_factors is not None:
                common_factors.append((index_key, my_common_factors))

            # }}}
        else:
            # {{{ match, filter existing common factors

            _, my_common_factors = common_factors[cf_index]

            unif_subst_map = SubstitutionMapper(
                    make_subst_func(unif_result.lmap))

            for term in iterate_as(Sum, rhs):
                if term == lhs:
                    continue

                for part in iterate_as(Product, term):
                    if var_name in get_dependencies(part):
                        raise LoopyError("unexpected dependency on '%s' "
                                "in RHS of instruction '%s'"
                                % (var_name, insn.id))

                product_parts = set(iterate_as(Product, term))

                my_common_factors = {
                        cf for cf in my_common_factors
                        if unif_subst_map(cf) in product_parts}

            common_factors[cf_index] = (index_key, my_common_factors)

            # }}}

    # }}}

    common_factors = [
            (ik, cf) for ik, cf in common_factors
            if cf]

    if not common_factors:
        raise LoopyError("no common factors found")

    # {{{ remove common factors

    new_insns = []

    for insn in kernel.instructions:
        if not isinstance(insn, Assignment) or not is_assignee(insn):
            new_insns.append(insn)
            continue

        index_key = extract_index_key(insn.assignee)

        lhs = insn.assignee
        rhs = insn.expression

        if is_zero(rhs):
            new_insns.append(insn)
            continue

        index_key = extract_index_key(lhs)
        cf_index, unif_result = find_unifiable_cf_index(index_key)

        if cf_index is None:
            new_insns.append(insn)
            continue

        _, my_common_factors = common_factors[cf_index]

        unif_subst_map = SubstitutionMapper(
                make_subst_func(unif_result.lmap))

        mapped_my_common_factors = {
                unif_subst_map(cf)
                for cf in my_common_factors}

        new_sum_terms = []

        for term in iterate_as(Sum, rhs):
            if term == lhs:
                new_sum_terms.append(term)
                continue

            new_sum_terms.append(
                    flattened_product([
                        part
                        for part in iterate_as(Product, term)
                        if part not in mapped_my_common_factors
                        ]))

        new_insns.append(
                insn.copy(expression=flattened_sum(new_sum_terms)))

    # }}}

    # {{{ substitute common factors into usage sites

    def find_substitution(expr):
        if isinstance(expr, Subscript):
            v = expr.aggregate.name
        elif isinstance(expr, Variable):
            v = expr.name
        else:
            return expr

        if v != var_name:
            return expr

        index_key = extract_index_key(expr)
        cf_index, unif_result = find_unifiable_cf_index(index_key)

        unif_subst_map = SubstitutionMapper(
                make_subst_func(unif_result.lmap))

        _, my_common_factors = common_factors[cf_index]

        if my_common_factors is not None:
            return flattened_product(
                    [unif_subst_map(cf) for cf in my_common_factors]
                    + [expr])
        else:
            return expr

    insns = new_insns
    new_insns = []

    subm = SubstitutionMapper(find_substitution)

    for insn in insns:
        if not isinstance(insn, Assignment) or is_assignee(insn):
            new_insns.append(insn)
            continue

        new_insns.append(insn.with_transformed_expressions(subm))

    # }}}

    return kernel.copy(instructions=new_insns)

# }}}


# vim: foldmethod=marker

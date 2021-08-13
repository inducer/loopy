__copyright__ = "Copyright (C) 2019 James Stevens"

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
dim_type = isl.dim_type


def prettier_map_string(map_obj):
    return str(map_obj
               ).replace("{ ", "{\n").replace(" }", "\n}").replace("; ", ";\n")


def insert_and_name_isl_dims(isl_set, dt, names, new_idx_start):
    new_set = isl_set.insert_dims(dt, new_idx_start, len(names))
    for i, name in enumerate(names):
        new_set = new_set.set_dim_name(dt, new_idx_start+i, name)
    return new_set


def add_and_name_isl_dims(isl_map, dt, names):
    new_idx_start = isl_map.dim(dt)
    new_map = isl_map.add_dims(dt, len(names))
    for i, name in enumerate(names):
        new_map = new_map.set_dim_name(dt, new_idx_start+i, name)
    return new_map


def reorder_dims_by_name(
        isl_set, dt, desired_dims_ordered):
    """Return an isl_set with the dimensions of the specified dim type
    in the specified order.

    :arg isl_set: A :class:`islpy.Set` whose dimensions are
        to be reordered.

    :arg dt: A :class:`islpy.dim_type`, i.e., an :class:`int`,
        specifying the dimension to be reordered.

    :arg desired_dims_ordered: A :class:`list` of :class:`str` elements
        representing the desired dimensions in order by dimension name.

    :returns: An :class:`islpy.Set` matching `isl_set` with the
        dimension order matching `desired_dims_ordered`.

    """

    assert dt != dim_type.param
    assert set(isl_set.get_var_names(dt)) == set(desired_dims_ordered)

    other_dt = dim_type.param
    other_dim_len = len(isl_set.get_var_names(other_dt))

    new_set = isl_set.copy()
    for desired_idx, name in enumerate(desired_dims_ordered):

        current_idx = new_set.find_dim_by_name(dt, name)
        if current_idx != desired_idx:
            # First move to other dim because isl is stupid
            new_set = new_set.move_dims(
                other_dt, other_dim_len, dt, current_idx, 1)
            # Now move it where we actually want it
            new_set = new_set.move_dims(
                dt, desired_idx, other_dt, other_dim_len, 1)

    return new_set


def ensure_dim_names_match_and_align(obj_map, tgt_map):

    # first make sure names match
    if not all(
            set(obj_map.get_var_names(dt)) == set(tgt_map.get_var_names(dt))
            for dt in
            [dim_type.in_, dim_type.out, dim_type.param]):
        raise ValueError(
            "Cannot align spaces; names don't match:\n%s\n%s"
            % (prettier_map_string(obj_map), prettier_map_string(tgt_map))
            )

    return isl.align_spaces(obj_map, tgt_map)


def add_eq_isl_constraint_from_names(isl_map, var1, var2):
    # add constraint var1 = var2
    return isl_map.add_constraint(
               isl.Constraint.eq_from_names(
                   isl_map.space,
                   {1: 0, var1: 1, var2: -1}))


def append_mark_to_isl_map_var_names(old_isl_map, dt, mark):
    """Return an :class:`islpy.Map` with a mark appended to the specified
    dimension names.

    :arg old_isl_map: An :class:`islpy.Map`.

    :arg dt: An :class:`islpy.dim_type`, i.e., an :class:`int`,
        specifying the dimension to be marked.

    :arg mark: A :class:`str` to be appended to the specified dimension
        names. If not provided, `mark` defaults to an apostrophe.

    :returns: An :class:`islpy.Map` matching `old_isl_map` with
        `mark` appended to the `dt` dimension names.

    """

    new_map = old_isl_map.copy()
    for i in range(len(old_isl_map.get_var_names(dt))):
        new_map = new_map.set_dim_name(dt, i, old_isl_map.get_dim_name(
            dt, i)+mark)
    return new_map


def append_mark_to_strings(strings, mark):
    return [s+mark for s in strings]


# {{{ make_dep_map

def make_dep_map(s, self_dep=False, knl_with_domains=None):
    """Given a string representation of a before->after mapping of statement
    instances, create an :class:`islpy.Map` representing the dependency. Insert
    a dimension into this map to represent the statement identifier for both
    the 'before' and 'after' statements. If ``knl_with_domains`` is provided,
    intersect the input and output map domains with the domains for the
    inames found in the kernel.

    :arg s: An :class:`str` describing a before->after mapping of statement
        instances using islpy map syntax. The input and output spaces
        of the map represented by this string should *not* include a dimension
        for statement identifiers; these dimension will be added. Inames in
        the input space should be suffixed with
        ``loopy.schedule.checker.schedule.BEFORE_MARK``.

    :arg self_dep: A :class`bool` expressing whether the depender and
        dependee are the same instruction. If so, the value for *both* the
        input and output statement identifier dimensions will be set to 0.
        If not, the value for the *output* statement identifier dimension will
        be set to 1.

    :arg knl_with_domains: A :class:`loopy.kernel.LoopKernel` containing iname
        domains that will be used to constrain the inames in the dependency map.
        If provided, the domains for the inames found in the dependency will be
        intersected with their domains expressed in the kernel.

    :returns: An :class:`islpy.Map` representing a dependency as a mapping from
        from each instance of the first statement to all instances of the
        second statement that must occur later.

    """

    from loopy.schedule.checker.schedule import (
        BEFORE_MARK,
        STATEMENT_VAR_NAME,
    )

    # Pass the input string to isl.Map to initialize the map
    map_init = isl.Map(s)

    # {{{ Islpy drops apostrophes, make sure this hasn't changed
    # and manually add the mark if necessary

    if BEFORE_MARK == "'":
        for dim_name in map_init.get_var_names(dim_type.in_):
            assert BEFORE_MARK not in dim_name

        # Append BEFORE_MARK to in_ dims
        map_marked = append_mark_to_isl_map_var_names(
            map_init, dim_type.in_, BEFORE_MARK)

    # }}}

    # {{{ Insert input/output statement dims and set them to 0 or 1

    map_with_stmts = insert_and_name_isl_dims(
        map_marked, dim_type.in_, [STATEMENT_VAR_NAME+BEFORE_MARK], 0)
    map_with_stmts = insert_and_name_isl_dims(
        map_with_stmts, dim_type.out, [STATEMENT_VAR_NAME], 0)

    sid_after = 0 if self_dep else 1

    map_with_stmts = map_with_stmts.add_constraint(
        isl.Constraint.eq_from_names(
            map_with_stmts.space,
            {1: 0, STATEMENT_VAR_NAME+BEFORE_MARK: -1}))

    map_with_stmts = map_with_stmts.add_constraint(
        isl.Constraint.eq_from_names(
            map_with_stmts.space,
            {1: sid_after, STATEMENT_VAR_NAME: -1}))

    # }}}

    # {{{ Intersect map domain and range with iname domains in knl

    if knl_with_domains is not None:

        if BEFORE_MARK != "'":
            raise NotImplementedError(
                "make_dep_map() does not yet handle a knl_with_domains argument "
                "when BEFORE_MARK != \"'\"")

        # {{{ Get inames domain for input and output inames

        # Get the inames from map_init; islpy already dropped the apostrophes
        inames_in = map_init.get_var_names(dim_type.in_)
        inames_out = map_init.get_var_names(dim_type.out)

        # Get inames domain
        inames_in_dom = knl_with_domains.get_inames_domain(
            inames_in).project_out_except(inames_in, [dim_type.set])
        inames_out_dom = knl_with_domains.get_inames_domain(
            inames_out).project_out_except(inames_out, [dim_type.set])

        # Mark dependee inames
        inames_in_dom_marked = append_mark_to_isl_map_var_names(
            inames_in_dom, dim_type.set, BEFORE_MARK)

        # }}}

        # {{{ Align spaces for iname domains with dep map (which adds the stmt var)

        inames_in_dom_marked_aligned = isl.align_spaces(
            inames_in_dom_marked, map_with_stmts.domain(),
            obj_bigger_ok=True)  # e.g., params might exist
        inames_out_dom_aligned = isl.align_spaces(
            inames_out_dom, map_with_stmts.range(),
            obj_bigger_ok=True)  # e.g., params might exist

        # }}}

        # Intersect iname domains with dependency map
        map_with_stmts = map_with_stmts.intersect_range(
            inames_out_dom_aligned
            ).intersect_domain(inames_in_dom_marked_aligned)

    # }}}

    return map_with_stmts

# }}}


def sorted_union_of_names_in_isl_sets(
        isl_sets,
        set_dim=dim_type.set):
    r"""Return a sorted list of the union of all variable names found in
    the provided :class:`islpy.Set`\ s.
    """

    inames = set().union(*[isl_set.get_var_names(set_dim) for isl_set in isl_sets])

    # Sorting is not necessary, but keeps results consistent between runs
    return sorted(inames)


def convert_map_to_set(isl_map):
    # (also works for spaces)
    n_in_dims = len(isl_map.get_var_names(dim_type.in_))
    n_out_dims = len(isl_map.get_var_names(dim_type.out))
    return isl_map.move_dims(
        dim_type.in_, n_in_dims, dim_type.out, 0, n_out_dims
        ).domain(), n_in_dims, n_out_dims


def convert_set_back_to_map(isl_set, n_old_in_dims, n_old_out_dims):
    return isl.Map.from_domain(
        isl_set).move_dims(
            dim_type.out, 0, dim_type.in_, n_old_in_dims, n_old_out_dims)


def create_symbolic_map_from_tuples(
        tuple_pairs_with_domains,
        space,
        ):
    """Return an :class:`islpy.Map` constructed using the provided space,
    mapping input->output tuples provided in `tuple_pairs_with_domains`,
    with each set of tuple variables constrained by the domains provided.

    :arg tuple_pairs_with_domains: A :class:`list` with each element being
        a tuple of the form `((tup_in, tup_out), domain)`.
        `tup_in` and `tup_out` are tuples containing elements of type
        :class:`int` and :class:`str` representing values for the
        input and output dimensions in `space`, and `domain` is a
        :class:`islpy.Set` constraining variable bounds.

    :arg space: A :class:`islpy.Space` to be used to create the map.

    :returns: A :class:`islpy.Map` constructed using the provided space
        as follows. For each `((tup_in, tup_out), domain)` in
        `tuple_pairs_with_domains`, map
        `(tup_in)->(tup_out) : domain`, where `tup_in` and `tup_out` are
        numeric or symbolic values assigned to the input and output
        dimension variables in `space`, and `domain` specifies conditions
        on these values.

    """
    # FIXME allow None for domains

    space_out_names = space.get_var_names(dim_type.out)
    space_in_names = space.get_var_names(dim_type.in_)

    def _conjunction_of_dim_eq_conditions(dim_names, values, var_name_to_pwaff):
        condition = var_name_to_pwaff[0].eq_set(var_name_to_pwaff[0])
        for dim_name, val in zip(dim_names, values):
            if isinstance(val, int):
                condition = condition \
                    & var_name_to_pwaff[dim_name].eq_set(var_name_to_pwaff[0]+val)
            else:
                condition = condition \
                    & var_name_to_pwaff[dim_name].eq_set(var_name_to_pwaff[val])
        return condition

    # Get islvars from space
    var_name_to_pwaff = isl.affs_from_space(
        space.move_dims(
            dim_type.out, 0,
            dim_type.in_, 0,
            len(space_in_names),
            ).range()
        )

    # Initialize union of maps to empty
    union_of_maps = isl.Map.from_domain(
        var_name_to_pwaff[0].eq_set(var_name_to_pwaff[0]+1)  # 0 == 1 (false)
        ).move_dims(
            dim_type.out, 0, dim_type.in_, len(space_in_names), len(space_out_names))

    # Loop through tuple pairs
    for (tup_in, tup_out), dom in tuple_pairs_with_domains:

        # Set values for 'in' dimension using tuple vals
        condition = _conjunction_of_dim_eq_conditions(
            space_in_names, tup_in, var_name_to_pwaff)

        # Set values for 'out' dimension using tuple vals
        condition = condition & _conjunction_of_dim_eq_conditions(
            space_out_names, tup_out, var_name_to_pwaff)

        # Convert set to map by moving dimensions around
        map_from_set = isl.Map.from_domain(condition)
        map_from_set = map_from_set.move_dims(
            dim_type.out, 0, dim_type.in_,
            len(space_in_names), len(space_out_names))

        # Align the *out* dims of dom with the space *in_* dims
        # in preparation for intersection
        dom_with_set_dim_aligned = reorder_dims_by_name(
            dom, dim_type.set,
            space_in_names,
            )

        # Intersect domain with this map
        union_of_maps = union_of_maps.union(
            map_from_set.intersect_domain(dom_with_set_dim_aligned))

    return union_of_maps


def partition_inames_by_concurrency(knl):
    from loopy.kernel.data import ConcurrentTag
    conc_inames = set()
    non_conc_inames = set()

    all_inames = knl.all_inames()
    for iname in all_inames:
        if knl.iname_tags_of_type(iname, ConcurrentTag):
            conc_inames.add(iname)
        else:
            non_conc_inames.add(iname)

    return conc_inames, all_inames-conc_inames


def get_EnterLoop_inames(linearization_items):
    from loopy.schedule import EnterLoop

    # Note: each iname must live in len-1 list to avoid char separation
    return set().union(*[
        [item.iname, ] for item in linearization_items
        if isinstance(item, EnterLoop)
        ])


def create_elementwise_comparison_conjunction_set(
        names0, names1, var_name_to_pwaff, op="eq"):
    """Create a set constrained by the conjunction of conditions comparing
    `names0` to `names1`.

    :arg names0: A list of :class:`str` representing variable names.

    :arg names1: A list of :class:`str` representing variable names.

    :arg var_name_to_pwaff: A dictionary from variable names to :class:`islpy.PwAff`
        instances that represent each of the variables
        (var_name_to_pwaff may be produced by `islpy.make_zero_and_vars`). The key
        '0' is also include and represents a :class:`islpy.PwAff` zero constant.

    :arg op: A :class:`str` describing the operator to use when creating
        the set constraints. Options: `eq` for `=`, `lt` for `<`

    :returns: A set involving `var_name_to_pwaff` cosntrained by the constraints
        `{names0[0] <op> names1[0] and names0[1] <op> names1[1] and ...}`.

    """

    # initialize set with constraint that is always true
    conj_set = var_name_to_pwaff[0].eq_set(var_name_to_pwaff[0])
    for n0, n1 in zip(names0, names1):
        if op == "eq":
            conj_set = conj_set & var_name_to_pwaff[n0].eq_set(var_name_to_pwaff[n1])
        elif op == "ne":
            conj_set = conj_set & var_name_to_pwaff[n0].ne_set(var_name_to_pwaff[n1])
        elif op == "lt":
            conj_set = conj_set & var_name_to_pwaff[n0].lt_set(var_name_to_pwaff[n1])

    return conj_set

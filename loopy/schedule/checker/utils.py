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
    return str(
        map_obj
        ).replace("{ ", "{\n").replace(" }", "\n}").replace("; ", ";\n").replace(
        "(", "\n (")


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


def move_dims_by_name(
        isl_obj, dst_type, dst_pos_start, src_type, dim_names):
    dst_pos = dst_pos_start
    for dim_name in dim_names:
        src_idx = isl_obj.find_dim_by_name(src_type, dim_name)
        if src_idx == -1:
            raise ValueError(
                "move_dims_by_name did not find dimension %s"
                % (dim_name))
        isl_obj = isl_obj.move_dims(
            dst_type, dst_pos, src_type, src_idx, 1)
        dst_pos += 1
    return isl_obj


def remove_dims_by_name(isl_obj, dt, dim_names):
    for dim_name in dim_names:
        idx = isl_obj.find_dim_by_name(dt, dim_name)
        if idx == -1:
            raise ValueError(
                "remove_dims_by_name did not find dimension %s"
                % (dim_name))
        isl_obj = isl_obj.remove_dims(dt, idx, 1)
    return isl_obj


def rename_dims(
        isl_set, rename_map,
        dts=(dim_type.in_, dim_type.out, dim_type.param)):
    new_isl_set = isl_set.copy()
    for dt in dts:
        for idx, old_name in enumerate(isl_set.get_var_names(dt)):
            if old_name in rename_map:
                new_isl_set = new_isl_set.set_dim_name(
                    dt, idx, rename_map[old_name])
    return new_isl_set


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
    assert isinstance(var1, str)
    # var2 may be an int or a string
    if isinstance(var2, str):
        return isl_map.add_constraint(
                   isl.Constraint.eq_from_names(
                       isl_map.space,
                       {1: 0, var1: 1, var2: -1}))
    else:
        assert isinstance(var2, int)
        return isl_map.add_constraint(
                   isl.Constraint.eq_from_names(
                       isl_map.space,
                       {1: var2, var1: -1}))


def add_int_bounds_to_isl_var(isl_map, var, lbound, ubound):
    # NOTE: these are inclusive bounds
    # add constraint var1 = var2
    return isl_map.add_constraint(
        isl.Constraint.ineq_from_names(
            isl_map.space, {1: -1*lbound, var: 1})
        ).add_constraint(
            isl.Constraint.ineq_from_names(
                isl_map.space, {1: ubound, var: -1}))


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
    assert isinstance(strings, list)
    return [s+mark for s in strings]


def sorted_union_of_names_in_isl_sets(
        isl_sets,
        set_dim=dim_type.set):
    r"""Return a sorted list of the union of all variable names found in
    the provided :class:`islpy.Set`\ s.
    """

    inames = set().union(*[isl_set.get_var_names(set_dim) for isl_set in isl_sets])

    # Sorting is not necessary, but keeps results consistent between runs
    return sorted(inames)


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

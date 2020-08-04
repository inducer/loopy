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


def prettier_map_string(map_obj):
    return str(map_obj
               ).replace("{ ", "{\n").replace(" }", "\n}").replace("; ", ";\n")


def add_dims_to_isl_set(isl_set, dim_type, names, new_idx_start):
    new_set = isl_set.insert_dims(
        dim_type, new_idx_start, len(names)
        ).set_dim_name(dim_type, new_idx_start, names[0])
    for i, name in enumerate(names[1:]):
        new_set = new_set.set_dim_name(dim_type, new_idx_start+1+i, name)
    return new_set


def map_names_match_check(
        obj_map,
        desired_names,
        dim_type,
        assert_subset=True,
        assert_permutation=True,
        ):
    """Raise an error if names of the specified map dimension do not match
    the desired names
    """

    obj_map_names = obj_map.space.get_var_names(dim_type)
    if assert_permutation:
        if not set(obj_map_names) == set(desired_names):
            raise ValueError(
                "Set of map names %s for dim %s does not match target set %s"
                % (obj_map_names, dim_type, desired_names))
    elif assert_subset:
        if not set(obj_map_names).issubset(desired_names):
            raise ValueError(
                "Map names %s for dim %s are not a subset of target names %s"
                % (obj_map_names, dim_type, desired_names))


def insert_missing_dims_and_reorder_by_name(
        isl_set, dim_type, desired_dims_ordered):
    """Return an isl_set with the dimensions in the specified order.

    :arg isl_set: A :class:`islpy.Set` whose dimensions are
        to be reordered and, if necessary, augmented with missing dimensions.

    :arg dim_type: A :class:`islpy.dim_type`, i.e., an :class:`int`,
        specifying the dimension to be reordered.

    :arg desired_dims_ordered: A :class:`list` of :class:`str` elements
        representing the desired dimensions in order by dimension name.

    :returns: An :class:`islpy.Set` matching `isl_set` with the
        dimension order matching `desired_dims_ordered`,
        including additional dimensions present in `desred_dims_ordered`
        that are not present in `isl_set`.

    """

    map_names_match_check(
        isl_set, desired_dims_ordered, dim_type,
        assert_subset=True, assert_permutation=False)

    assert dim_type != isl.dim_type.param

    other_dim_type = isl.dim_type.param
    other_dim_len = len(isl_set.get_var_names(other_dim_type))

    new_set = isl_set.copy()
    for desired_idx, name in enumerate(desired_dims_ordered):
        # If iname doesn't exist in set, add dim
        if name not in new_set.get_var_names(dim_type):
            # Insert missing dim in correct location
            new_set = new_set.insert_dims(
                dim_type, desired_idx, 1
                ).set_dim_name(dim_type, desired_idx, name)
        else:  # Iname exists in set
            current_idx = new_set.find_dim_by_name(dim_type, name)
            if current_idx != desired_idx:
                # First move to other dim because isl is stupid
                new_set = new_set.move_dims(
                    other_dim_type, other_dim_len, dim_type, current_idx, 1)
                # Now move it where we actually want it
                new_set = new_set.move_dims(
                    dim_type, desired_idx, other_dim_type, other_dim_len, 1)

    return new_set


def ensure_dim_names_match_and_align(obj_map, tgt_map):

    # first make sure names match
    for dt in [isl.dim_type.in_, isl.dim_type.out, isl.dim_type.param]:
        map_names_match_check(
            obj_map, tgt_map.get_var_names(dt), dt,
            assert_permutation=True)

    aligned_obj_map = isl.align_spaces(obj_map, tgt_map)

    return aligned_obj_map


def sorted_union_of_names_in_isl_sets(
        isl_sets,
        set_dim=isl.dim_type.set):
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
    # TODO allow None for domains

    dim_type = isl.dim_type

    space_out_names = space.get_var_names(dim_type.out)
    space_in_names = space.get_var_names(isl.dim_type.in_)

    # Get islvars from space
    islvars = isl.affs_from_space(
        space.move_dims(
            isl.dim_type.out, 0,
            isl.dim_type.in_, 0,
            len(space_in_names),
            ).range()
        )

    def _conjunction_of_dim_eq_conditions(dim_names, values, islvars):
        condition = islvars[0].eq_set(islvars[0])
        for dim_name, val in zip(dim_names, values):
            if isinstance(val, int):
                condition = condition \
                    & islvars[dim_name].eq_set(islvars[0]+val)
            else:
                condition = condition \
                    & islvars[dim_name].eq_set(islvars[val])
        return condition

    # Initialize union of maps to empty
    union_of_maps = isl.Map.from_domain(
        islvars[0].eq_set(islvars[0]+1)  # 0 == 1 (false)
        ).move_dims(
            dim_type.out, 0, dim_type.in_, len(space_in_names), len(space_out_names))

    # Loop through tuple pairs
    for (tup_in, tup_out), dom in tuple_pairs_with_domains:

        # Set values for 'in' dimension using tuple vals
        condition = _conjunction_of_dim_eq_conditions(
            space_in_names, tup_in, islvars)

        # Set values for 'out' dimension using tuple vals
        condition = condition & _conjunction_of_dim_eq_conditions(
            space_out_names, tup_out, islvars)

        # Convert set to map by moving dimensions around
        map_from_set = isl.Map.from_domain(condition)
        map_from_set = map_from_set.move_dims(
            dim_type.out, 0, dim_type.in_,
            len(space_in_names), len(space_out_names))

        assert space_in_names == map_from_set.get_var_names(
            isl.dim_type.in_)

        # If there are any dimensions in dom that are missing from
        # map_from_set, we have a problem I think?
        # (assertion checks this in add_missing...)
        dom_with_all_inames = insert_missing_dims_and_reorder_by_name(
            dom, isl.dim_type.set,
            space_in_names,
            )

        # Intersect domain with this map
        union_of_maps = union_of_maps.union(
            map_from_set.intersect_domain(dom_with_all_inames))

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


def get_insn_id_from_linearization_item(linearization_item):
    from loopy.schedule import Barrier
    if isinstance(linearization_item, Barrier):
        return linearization_item.originating_insn_id
    else:
        return linearization_item.insn_id


def get_EnterLoop_inames(linearization_items):
    from loopy.schedule import EnterLoop
    loop_inames = set()
    for linearization_item in linearization_items:
        if isinstance(linearization_item, EnterLoop):
            loop_inames.add(linearization_item.iname)
    return loop_inames

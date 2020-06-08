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


def get_islvars_from_space(space):
    param_names = space.get_var_names(isl.dim_type.param)
    in_names = space.get_var_names(isl.dim_type.in_)
    out_names = space.get_var_names(isl.dim_type.out)
    return isl.make_zero_and_vars(in_names+out_names, param_names)


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


def reorder_dims_by_name(
        isl_set, dim_type, desired_dims_ordered,
        add_missing=False, new_names_are_permutation_only=False):
    """Return an isl_set with the dimensions in the specified order.

    :arg isl_set: A :class:`islpy.Set` whose dimensions are
        to be reordered.

    :arg dim_type: A :class:`islpy.dim_type`, i.e., an :class:`int`,
        specifying the dimension to be reordered.

    :arg desired_dims_ordered: A :class:`list` of :class:`str` elements
        representing the desired dimensions order by dimension name.

    :arg add_missing: A :class:`bool` specifying whether to insert
        dimensions (by name) found in `desired_dims_ordered` that are not
        present in `isl_set`.

    :arg new_names_are_permutation_only: A :class:`bool` indicating that
        `desired_dims_ordered` contains the same names as the specified
        dimensions in `isl_set`, and does not, e.g., contain additional
        dimension names not found in `isl_set`. If set to True, and these
        two sets of names do not match, an error is produced.

    :returns: An :class:`islpy.Set` matching `isl_set` with the
        dimension order matching `desired_dims_ordered`, optionally
        including additional dimensions present in `desred_dims_ordered`
        that are not present in `isl_set`.

    """

    map_names_match_check(
        isl_set, desired_dims_ordered, dim_type,
        assert_subset=True, assert_permutation=False)

    assert dim_type != isl.dim_type.param

    if new_names_are_permutation_only and (
            set(isl_set.get_var_names(dim_type))
            != set(desired_dims_ordered)):
        raise ValueError(
            "Var name sets must match with new_names_are_permutation_only=True. "
            "isl vars: %s, desired dims: %s"
            % (isl_set.get_var_names(dim_type), desired_dims_ordered))

    other_dim_type = isl.dim_type.param
    other_dim_len = len(isl_set.get_var_names(other_dim_type))

    new_set = isl_set.copy()
    for desired_idx, name in enumerate(desired_dims_ordered):
        # if iname doesn't exist in set, add dim:
        if name not in new_set.get_var_names(dim_type):
            if add_missing:
                # insert missing dim in correct location
                new_set = new_set.insert_dims(
                    dim_type, desired_idx, 1
                    ).set_dim_name(
                    dim_type, desired_idx, name)
        else:  # iname exists in set
            current_idx = new_set.find_dim_by_name(dim_type, name)
            if current_idx != desired_idx:
                # move_dims(dst_type, dst_idx, src_type, src_idx, n)

                # first move to other dim because isl is stupid
                new_set = new_set.move_dims(
                    other_dim_type, other_dim_len, dim_type, current_idx, 1)

                # now move it where we actually want it
                new_set = new_set.move_dims(
                    dim_type, desired_idx, other_dim_type, other_dim_len, 1)

    return new_set


def align_isl_maps_by_var_names(input_map, target_map):

    # first make sure names match
    for dt in [isl.dim_type.in_, isl.dim_type.out, isl.dim_type.param]:
        map_names_match_check(
            input_map, target_map.get_var_names(dt), dt,
            assert_permutation=True)

    aligned_input_map = isl.align_spaces(input_map, target_map)

    # TODO remove once satisfied that above can replace below:

    # align params
    _aligned_input_map = input_map.align_params(target_map.space)

    # align in_ dims
    target_map_in_names = target_map.space.get_var_names(isl.dim_type.in_)
    _aligned_input_map = reorder_dims_by_name(
        _aligned_input_map,
        isl.dim_type.in_,
        target_map_in_names,
        add_missing=False,
        new_names_are_permutation_only=True,
        )

    # align out dims
    target_map_out_names = target_map.space.get_var_names(isl.dim_type.out)
    _aligned_input_map = reorder_dims_by_name(
        _aligned_input_map,
        isl.dim_type.out,
        target_map_out_names,
        add_missing=False,
        new_names_are_permutation_only=True,
        )

    assert aligned_input_map == _aligned_input_map
    assert aligned_input_map.get_var_names(
        isl.dim_type.param) == _aligned_input_map.get_var_names(isl.dim_type.param)
    assert aligned_input_map.get_var_names(
        isl.dim_type.in_) == _aligned_input_map.get_var_names(isl.dim_type.in_)
    assert aligned_input_map.get_var_names(
        isl.dim_type.out) == _aligned_input_map.get_var_names(isl.dim_type.out)

    return aligned_input_map


def append_marker_to_isl_map_var_names(old_isl_map, dim_type, marker="'"):
    """Return an isl_map with marker appended to
        dim_type dimension names.

    :arg old_isl_map: A :class:`islpy.Map`.

    :arg dim_type: A :class:`islpy.dim_type`, i.e., an :class:`int`,
        specifying the dimension to be marked.

    :returns: A :class:`islpy.Map` matching `old_isl_map` with
        apostrophes appended to dim_type dimension names.

    """

    new_map = old_isl_map.copy()
    for i in range(len(old_isl_map.get_var_names(dim_type))):
        new_map = new_map.set_dim_name(dim_type, i, old_isl_map.get_dim_name(
            dim_type, i)+marker)
    return new_map


def append_marker_to_strings(strings, marker="'"):
    if not isinstance(strings, list):
        raise ValueError("append_marker_to_strings did not receive a list")
    else:
        return [s+marker for s in strings]


def _get_union(list_items):
    union = list_items[0]
    for s in list_items[1:]:
        union = union.union(s)
    return union


def list_var_names_in_isl_sets(
        isl_sets,
        set_dim=isl.dim_type.set):
    inames = set()
    for isl_set in isl_sets:
        inames.update(isl_set.get_var_names(set_dim))

    # sorting is not necessary, but keeps results consistent between runs
    return sorted(list(inames))


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
        dimension variables in `space`, and `domain` specifies constraints
        on these values.

    """
    # TODO allow None for domains

    dim_type = isl.dim_type

    space_out_names = space.get_var_names(dim_type.out)
    space_in_names = space.get_var_names(isl.dim_type.in_)

    islvars = get_islvars_from_space(space)

    # loop through pairs and create a set that will later be converted to a map

    all_maps = []
    for (tup_in, tup_out), dom in tuple_pairs_with_domains:

        # initialize constraint with true
        constraint = islvars[0].eq_set(islvars[0])

        # set values for 'in' dimension using tuple vals
        assert len(tup_in) == len(space_in_names)
        for dim_name, val_in in zip(space_in_names, tup_in):
            if isinstance(val_in, int):
                constraint = constraint \
                    & islvars[dim_name].eq_set(islvars[0]+val_in)
            else:
                constraint = constraint \
                    & islvars[dim_name].eq_set(islvars[val_in])

        # set values for 'out' dimension using tuple vals
        assert len(tup_out) == len(space_out_names)
        for dim_name, val_out in zip(space_out_names, tup_out):
            if isinstance(val_out, int):
                constraint = constraint \
                    & islvars[dim_name].eq_set(islvars[0]+val_out)
            else:
                constraint = constraint \
                    & islvars[dim_name].eq_set(islvars[val_out])

        # convert set to map by moving dimensions around
        map_from_set = isl.Map.from_domain(constraint)
        map_from_set = map_from_set.move_dims(
            dim_type.out, 0, dim_type.in_,
            len(space_in_names), len(space_out_names))

        assert space_in_names == map_from_set.get_var_names(
            isl.dim_type.in_)

        # if there are any dimensions in dom that are missing from
        # map_from_set, we have a problem I think?
        # (assertion checks this in add_missing...
        dom_with_all_inames = reorder_dims_by_name(
            dom, isl.dim_type.set,
            space_in_names,
            add_missing=True,
            new_names_are_permutation_only=False,
            )

        # intersect domain with this map
        all_maps.append(
            map_from_set.intersect_domain(dom_with_all_inames))

    return _get_union(all_maps)


def set_all_isl_space_names(
        isl_space, param_names=None, in_names=None, out_names=None):
    """Return a copy of `isl_space` with the specified dimension names.
        If no names are provided, use `p0, p1, ...` for parameters,
        `i0, i1, ...`, for in_ dimensions, and `o0, o1, ...` for out
        dimensions.

    """

    new_space = isl_space.copy()
    dim_type = isl.dim_type
    if param_names:
        for i, p in enumerate(param_names):
            new_space = new_space.set_dim_name(dim_type.param, i, p)
    else:
        for i in range(len(isl_space.get_var_names(dim_type.param))):
            new_space = new_space.set_dim_name(dim_type.param, i, "p%d" % (i))
    if in_names:
        for i, p in enumerate(in_names):
            new_space = new_space.set_dim_name(dim_type.in_, i, p)
    else:
        for i in range(len(isl_space.get_var_names(dim_type.in_))):
            new_space = new_space.set_dim_name(dim_type.in_, i, "i%d" % (i))
    if out_names:
        for i, p in enumerate(out_names):
            new_space = new_space.set_dim_name(dim_type.out, i, p)
    else:
        for i in range(len(isl_space.get_var_names(dim_type.out))):
            new_space = new_space.set_dim_name(dim_type.out, i, "o%d" % (i))
    return new_space


def get_isl_space(param_names, in_names, out_names):
    """Return an :class:`islpy.Space` with the specified dimension names.
    """

    space = isl.Space.alloc(
        isl.DEFAULT_CONTEXT, len(param_names), len(in_names), len(out_names))
    return set_all_isl_space_names(
        space, param_names=param_names, in_names=in_names, out_names=out_names)


def get_concurrent_inames(knl):
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


def get_EnterLoop_inames(linearization_items, knl):
    from loopy.schedule import EnterLoop
    loop_inames = set()
    for linearization_item in linearization_items:
        if isinstance(linearization_item, EnterLoop):
            loop_inames.add(linearization_item.iname)
    return loop_inames

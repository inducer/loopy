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
dt = isl.dim_type


def prettier_map_string(map_obj):
    return str(map_obj
               ).replace("{ ", "{\n").replace(" }", "\n}").replace("; ", ";\n")


def insert_and_name_isl_dims(isl_set, dim_type, names, new_idx_start):
    new_set = isl_set.insert_dims(dim_type, new_idx_start, len(names))
    for i, name in enumerate(names):
        new_set = new_set.set_dim_name(dim_type, new_idx_start+i, name)
    return new_set


def add_and_name_isl_dims(isl_map, dim_type, names):
    new_idx_start = isl_map.dim(dim_type)
    new_map = isl_map.add_dims(dim_type, len(names))
    for i, name in enumerate(names):
        new_map = new_map.set_dim_name(dim_type, new_idx_start+i, name)
    return new_map


def reorder_dims_by_name(
        isl_set, dim_type, desired_dims_ordered):
    """Return an isl_set with the dimensions of the specified dim_type
    in the specified order.

    :arg isl_set: A :class:`islpy.Set` whose dimensions are
        to be reordered.

    :arg dim_type: A :class:`islpy.dim_type`, i.e., an :class:`int`,
        specifying the dimension to be reordered.

    :arg desired_dims_ordered: A :class:`list` of :class:`str` elements
        representing the desired dimensions in order by dimension name.

    :returns: An :class:`islpy.Set` matching `isl_set` with the
        dimension order matching `desired_dims_ordered`.

    """

    assert dim_type != dt.param
    assert set(isl_set.get_var_names(dim_type)) == set(desired_dims_ordered)

    other_dim_type = dt.param
    other_dim_len = len(isl_set.get_var_names(other_dim_type))

    new_set = isl_set.copy()
    for desired_idx, name in enumerate(desired_dims_ordered):

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
    if not all(
            set(obj_map.get_var_names(dt)) == set(tgt_map.get_var_names(dt))
            for dt in
            [dt.in_, dt.out, dt.param]):
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


def append_mark_to_isl_map_var_names(old_isl_map, dim_type, mark):
    """Return an :class:`islpy.Map` with a mark appended to the specified
    dimension names.

    :arg old_isl_map: An :class:`islpy.Map`.

    :arg dim_type: An :class:`islpy.dim_type`, i.e., an :class:`int`,
        specifying the dimension to be marked.

    :arg mark: A :class:`str` to be appended to the specified dimension
        names. If not provided, `mark` defaults to an apostrophe.

    :returns: An :class:`islpy.Map` matching `old_isl_map` with
        `mark` appended to the `dim_type` dimension names.

    """

    new_map = old_isl_map.copy()
    for i in range(len(old_isl_map.get_var_names(dim_type))):
        new_map = new_map.set_dim_name(dim_type, i, old_isl_map.get_dim_name(
            dim_type, i)+mark)
    return new_map


def make_islvars_with_mark(
        var_names_needing_mark, other_var_names, mark, param_names=[]):
    """Return a dictionary from variable and parameter names
        to :class:`islpy.PwAff` instances that represent each of
        the variables and parameters, appending mark to
        var_names_needing_mark.

    :arg var_names_needing_mark: A :class:`list` of :class:`str`
        elements representing variable names to have marks appended.

    :arg other_var_names: A :class:`list` of :class:`str`
        elements representing variable names to be included as-is.

    :arg param_names:  A :class:`list` of :class:`str` elements
        representing parameter names.

    :returns: A dictionary from variable names to :class:`islpy.PwAff`
        instances that represent each of the variables
        (islvars may be produced by `islpy.make_zero_and_vars`). The key
        '0' is also include and represents a :class:`islpy.PwAff` zero constant.

    """
    # TODO update docstring

    def append_mark(items, mark):
        new_items = []
        for item in items:
            new_items.append(item+mark)
        return new_items

    return isl.make_zero_and_vars(
            append_mark(var_names_needing_mark, mark)
            + other_var_names, param_names)


def append_mark_to_strings(strings, mark):
    return [s+mark for s in strings]


def sorted_union_of_names_in_isl_sets(
        isl_sets,
        set_dim=dt.set):
    r"""Return a sorted list of the union of all variable names found in
    the provided :class:`islpy.Set`\ s.
    """

    inames = set().union(*[isl_set.get_var_names(set_dim) for isl_set in isl_sets])

    # Sorting is not necessary, but keeps results consistent between runs
    return sorted(inames)


def convert_map_to_set(isl_map):
    n_in_dims = len(isl_map.get_var_names(dt.in_))
    n_out_dims = len(isl_map.get_var_names(dt.out))
    return isl_map.move_dims(
        dt.in_, n_in_dims, dt.out, 0, n_out_dims
        ).domain(), n_in_dims, n_out_dims


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
    # TODO clarify this with more comments
    # TODO allow None for domains

    space_out_names = space.get_var_names(dt.out)
    space_in_names = space.get_var_names(dt.in_)

    # Get islvars from space
    islvars = isl.affs_from_space(
        space.move_dims(
            dt.out, 0,
            dt.in_, 0,
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
            dt.out, 0, dt.in_, len(space_in_names), len(space_out_names))

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
            dt.out, 0, dt.in_,
            len(space_in_names), len(space_out_names))

        # Align the *out* dims of dom with the space *in_* dims
        # in preparation for intersection
        dom_with_set_dim_aligned = reorder_dims_by_name(
            dom, dt.set,
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


def get_loop_nesting_map(knl, inames=None):
    # Create a mapping from each iname to inames that must be
    # nested inside that iname, based on nesting constraints.
    # Once the new nest constraint semantics are added,
    # this function will change and may not be necessary.

    if inames is None:
        inames = knl.all_inames()

    nests_outside_map = {}
    for outside_iname in inames:
        nested_inside_inames = set()
        for p_tuple in knl.loop_priority:
            if outside_iname in p_tuple:
                nested_inside_inames.update(
                    p_tuple[p_tuple.index(outside_iname)+1:])
        nests_outside_map[outside_iname] = nested_inside_inames

    return nests_outside_map


def get_insn_id_from_linearization_item(linearization_item):
    from loopy.schedule import Barrier
    if isinstance(linearization_item, Barrier):
        return linearization_item.originating_insn_id
    else:
        return linearization_item.insn_id


def get_all_nonconcurrent_insn_iname_subsets(
        knl, exclude_empty=False, non_conc_inames=None):
    """Return a :class:`set` of every unique set of non-concurrent
        inames used in a single instruction in a :class:`loopy.LoopKernel`.

    :arg knl: A :class:`loopy.LoopKernel`.

    :arg exclude_empty: A :class:`bool` specifying whether to
        exclude the empty set.

    :arg non_conc_inames: A :class:`set` of non-concurrent inames
        which may be provided if already known.

    :returns: A :class:`set` of every unique set of non-concurrent
        inames used in any instruction in a :class:`loopy.LoopKernel`.

    """

    if non_conc_inames is None:
        _, non_conc_inames = partition_inames_by_concurrency(knl)

    iname_sets = set()
    for insn in knl.instructions:
        iname_sets.add(insn.within_inames & non_conc_inames)

    if exclude_empty:
        iname_sets.discard(frozenset())

    return iname_sets


def get_linearization_item_ids_within_inames(knl, inames):
    linearization_item_ids = set()
    for insn in knl.instructions:
        if inames.issubset(insn.within_inames):
            linearization_item_ids.add(insn.id)
    return linearization_item_ids


# TODO use yield to clean this up
# TODO use topological sort from loopy, then find longest path in dag
def _generate_orderings_starting_w_prefix(
        allowed_after_dict, orderings, required_length=None,
        start_prefix=(), return_first_found=False):
    # alowed_after_dict = {str: set(str)}
    # start prefix = tuple(str)
    # orderings = set
    if start_prefix:
        next_items = allowed_after_dict[start_prefix[-1]]-set(start_prefix)
    else:
        next_items = allowed_after_dict.keys()

    if required_length:
        if len(start_prefix) == required_length:
            orderings.add(start_prefix)
            if return_first_found:
                return
    else:
        orderings.add(start_prefix)
        if return_first_found:
            return

    # return if no more items left
    if not next_items:
        return

    for next_item in next_items:
        new_prefix = start_prefix + (next_item,)
        _generate_orderings_starting_w_prefix(
                allowed_after_dict,
                orderings,
                required_length=required_length,
                start_prefix=new_prefix,
                return_first_found=return_first_found,
                )
        if return_first_found and orderings:
            return
    return


def get_orderings_of_length_n(
        allowed_after_dict, required_length, return_first_found=False):
    """Return all orderings found in tree represented by `allowed_after_dict`.

    :arg allowed_after_dict: A :class:`dict` mapping each :class:`string`
        names to a :class:`set` of names that are allowed to come after
        that name.

    :arg required_length: A :class:`int` representing the length required
        for all orderings. Orderings not matching the required length will
        not be returned.

    :arg return_first_found: A :class:`bool` specifying whether to return
        the first valid ordering found.

    :returns: A :class:`set` of all orderings that are *explicitly* allowed
        by the tree represented by `allowed_after_dict`. I.e., if we know
        a->b and c->b, we don't know enough to return a->c->b. Note that
        if the set for a dict key is empty, nothing is allowed to come after.

    """

    orderings = set()
    _generate_orderings_starting_w_prefix(
        allowed_after_dict,
        orderings,
        required_length=required_length,
        start_prefix=(),
        return_first_found=return_first_found,
        )
    return orderings


def get_EnterLoop_inames(linearization_items):
    from loopy.schedule import EnterLoop

    # Note: each iname must live in len-1 list to avoid char separation
    return set().union(*[
        [item.iname, ] for item in linearization_items
        if isinstance(item, EnterLoop)
        ])


def create_elementwise_comparison_conjunction_set(
        names0, names1, islvars, op="eq"):
    """Create a set constrained by the conjunction of conditions comparing
       `names0` to `names1`.

    :arg names0: A list of :class:`str` representing variable names.

    :arg names1: A list of :class:`str` representing variable names.

    :arg islvars: A dictionary from variable names to :class:`islpy.PwAff`
        instances that represent each of the variables
        (islvars may be produced by `islpy.make_zero_and_vars`). The key
        '0' is also include and represents a :class:`islpy.PwAff` zero constant.

    :arg op: A :class:`str` describing the operator to use when creating
        the set constraints. Options: `eq` for `=`, `lt` for `<`

    :returns: A set involving `islvars` cosntrained by the constraints
        `{names0[0] <op> names1[0] and names0[1] <op> names1[1] and ...}`.

    """

    # initialize set with constraint that is always true
    conj_set = islvars[0].eq_set(islvars[0])
    for n0, n1 in zip(names0, names1):
        if op == "eq":
            conj_set = conj_set & islvars[n0].eq_set(islvars[n1])
        elif op == "ne":
            conj_set = conj_set & islvars[n0].ne_set(islvars[n1])
        elif op == "lt":
            conj_set = conj_set & islvars[n0].lt_set(islvars[n1])

    return conj_set

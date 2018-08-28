from __future__ import division, absolute_import

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

import six  # noqa
import islpy as isl

from loopy.diagnostic import LoopyError
from loopy.symbolic import CombineMapper


# {{{ find_instructions

def find_instructions(kernel, insn_match):
    from loopy.match import parse_match
    match = parse_match(insn_match)
    return [insn for insn in kernel.instructions if match(kernel, insn)]

# }}}


# {{{ map_instructions

def map_instructions(kernel, insn_match, f):
    from loopy.match import parse_match
    match = parse_match(insn_match)

    new_insns = []

    for insn in kernel.instructions:
        if match(kernel, insn):
            new_insns.append(f(insn))
        else:
            new_insns.append(insn)

    return kernel.copy(instructions=new_insns)

# }}}


# {{{ set_instruction_priority

def set_instruction_priority(kernel, insn_match, priority):
    """Set the priority of instructions matching *insn_match* to *priority*.

    *insn_match* may be any instruction id match understood by
    :func:`loopy.match.parse_match`.
    """

    def set_prio(insn):
        return insn.copy(priority=priority)

    return map_instructions(kernel, insn_match, set_prio)

# }}}


# {{{ add_dependency

def add_dependency(kernel, insn_match, depends_on):
    """Add the instruction dependency *dependency* to the instructions matched
    by *insn_match*.

    *insn_match* and *depends_on* may be any instruction id match understood by
    :func:`loopy.match.parse_match`.

    .. versionchanged:: 2016.3

        Third argument renamed to *depends_on* for clarity, allowed to
        be not just ID but also match expression.
    """

    if isinstance(depends_on, str) and depends_on in kernel.id_to_insn:
        added_deps = frozenset([depends_on])
    else:
        added_deps = frozenset(
                dep.id for dep in find_instructions(kernel, depends_on))

    if not added_deps:
        raise LoopyError("no instructions found matching '%s' "
                "(to add as dependencies)" % depends_on)

    matched = [False]

    def add_dep(insn):
        new_deps = insn.depends_on
        matched[0] = True
        if new_deps is None:
            new_deps = added_deps
        else:
            new_deps = new_deps | added_deps

        return insn.copy(depends_on=new_deps)

    result = map_instructions(kernel, insn_match, add_dep)

    if not matched[0]:
        raise LoopyError("no instructions found matching '%s' "
                "(to which dependencies would be added)" % insn_match)

    return result

# }}}


# {{{ remove_instructions

def remove_instructions(kernel, insn_ids):
    """Return a new kernel with instructions in *insn_ids* removed.

    Dependencies across (one, for now) deleted isntructions are propagated.
    Behavior is undefined for now for chains of dependencies within the
    set of deleted instructions.

    This also updates *no_sync_with* for all instructions.
    """

    if not insn_ids:
        return kernel

    assert isinstance(insn_ids, set)
    id_to_insn = kernel.id_to_insn

    new_insns = []
    for insn in kernel.instructions:
        if insn.id in insn_ids:
            continue

        # transitively propagate dependencies
        # (only one level for now)
        if insn.depends_on is None:
            depends_on = frozenset()
        else:
            depends_on = insn.depends_on

        new_deps = depends_on - insn_ids

        for dep_id in depends_on & insn_ids:
            new_deps = new_deps | id_to_insn[dep_id].depends_on

        # update no_sync_with

        new_no_sync_with = frozenset((insn_id, scope)
                for insn_id, scope in insn.no_sync_with
                if insn_id not in insn_ids)

        new_insns.append(
                insn.copy(depends_on=new_deps, no_sync_with=new_no_sync_with))

    return kernel.copy(
            instructions=new_insns)

# }}}


# {{{ replace_instruction_ids

def replace_instruction_ids(kernel, replacements):
    new_insns = []

    for insn in kernel.instructions:
        changed = False
        new_depends_on = []
        new_no_sync_with = []

        for dep in insn.depends_on:
            if dep in replacements:
                new_depends_on.extend(replacements[dep])
                changed = True
            else:
                new_depends_on.append(dep)

        for insn_id, scope in insn.no_sync_with:
            if insn_id in replacements:
                new_no_sync_with.extend(
                        (repl, scope) for repl in replacements[insn_id])
                changed = True
            else:
                new_no_sync_with.append((insn_id, scope))

        new_insns.append(
                insn.copy(
                    depends_on=frozenset(new_depends_on),
                    no_sync_with=frozenset(new_no_sync_with))
                if changed else insn)

    return kernel.copy(instructions=new_insns)

# }}}


# {{{ tag_instructions

def tag_instructions(kernel, new_tag, within=None):
    from loopy.match import parse_match
    within = parse_match(within)

    new_insns = []
    for insn in kernel.instructions:
        if within(kernel, insn):
            new_insns.append(
                    insn.copy(tags=insn.tags | frozenset([new_tag])))
        else:
            new_insns.append(insn)

    return kernel.copy(instructions=new_insns)

# }}}


# {{{ add nosync

def add_nosync(kernel, scope, source, sink, bidirectional=False, force=False,
        empty_ok=False):
    """Add a *no_sync_with* directive between *source* and *sink*.
    *no_sync_with* is only added if *sink* depends on *source* or
    if the instruction pair is in a conflicting group.

    This function does not check for the presence of a memory dependency.

    :arg kernel: The kernel
    :arg source: Either a single instruction id, or any instruction id
        match understood by :func:`loopy.match.parse_match`.
    :arg sink: Either a single instruction id, or any instruction id
        match understood by :func:`loopy.match.parse_match`.
    :arg scope: A valid *no_sync_with* scope. See
        :attr:`loopy.InstructionBase.no_sync_with` for allowable scopes.
    :arg bidirectional: A :class:`bool`. If *True*, add a *no_sync_with*
        to both the source and sink instructions, otherwise the directive
        is only added to the sink instructions.
    :arg force: A :class:`bool`. If *True*, add a *no_sync_with* directive
        even without the presence of a dependency edge or conflicting
        instruction group.
    :arg empty_ok: If *True*, do not complain even if no *nosync* tags were
        added as a result of the transformation.

    :return: The updated kernel

    .. versionchanged:: 2018.1

        If the transformation adds no *nosync* directives, it will complain.
        This used to silently pass. This behavior can be restored using
        *empty_ok*.
    """

    if isinstance(source, str) and source in kernel.id_to_insn:
        sources = frozenset([source])
    else:
        sources = frozenset(
                source.id for source in find_instructions(kernel, source))

    if isinstance(sink, str) and sink in kernel.id_to_insn:
        sinks = frozenset([sink])
    else:
        sinks = frozenset(
                sink.id for sink in find_instructions(kernel, sink))

    if not sources and not empty_ok:
        raise LoopyError("No match found for source specification '%s'." % source)
    if not sinks and not empty_ok:
        raise LoopyError("No match found for sink specification '%s'." % sink)

    def insns_in_conflicting_groups(insn1_id, insn2_id):
        insn1 = kernel.id_to_insn[insn1_id]
        insn2 = kernel.id_to_insn[insn2_id]
        return (
                bool(insn1.groups & insn2.conflicts_with_groups)
                or
                bool(insn2.groups & insn1.conflicts_with_groups))

    from collections import defaultdict
    nosync_to_add = defaultdict(set)

    rec_dep_map = kernel.recursive_insn_dep_map()
    for sink in sinks:
        for source in sources:

            needs_nosync = force or (
                    source in rec_dep_map[sink]
                    or insns_in_conflicting_groups(source, sink))

            if not needs_nosync:
                continue

            nosync_to_add[sink].add((source, scope))
            if bidirectional:
                nosync_to_add[source].add((sink, scope))

    if not nosync_to_add and not empty_ok:
        raise LoopyError("No nosync annotations were added as a result "
                "of this call. add_nosync will (by default) only add them to "
                "accompany existing depencies or group exclusions. Maybe you want "
                "to pass force=True?")

    new_instructions = list(kernel.instructions)

    for i, insn in enumerate(new_instructions):
        if insn.id in nosync_to_add:
            new_instructions[i] = insn.copy(no_sync_with=insn.no_sync_with
                    | frozenset(nosync_to_add[insn.id]))

    return kernel.copy(instructions=new_instructions)

# }}}


# {{{ uniquify_instruction_ids

def uniquify_instruction_ids(kernel):
    """Converts any ids that are :class:`loopy.UniqueName` or *None* into unique
    strings.

    This function does *not* deduplicate existing instruction ids.
    """

    from loopy.kernel.creation import UniqueName

    insn_ids = set(
            insn.id for insn in kernel.instructions
            if insn.id is not None and not isinstance(insn.id, UniqueName))

    from pytools import UniqueNameGenerator
    insn_id_gen = UniqueNameGenerator(insn_ids)

    new_instructions = []

    for insn in kernel.instructions:
        if insn.id is None:
            new_instructions.append(
                    insn.copy(id=insn_id_gen("insn")))
        elif isinstance(insn.id, UniqueName):
            new_instructions.append(
                    insn.copy(id=insn_id_gen(insn.id.name)))
        else:
            new_instructions.append(insn)

    return kernel.copy(instructions=new_instructions)

# }}}


# {{{ remove_work

class _MemAccessGatherer(CombineMapper):
    def __init__(self, kernel, address_space):
        self.kernel = kernel
        self.address_space = address_space

    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, expr):
        return set()

    def map_algebraic_leaf(self, expr):
        return set()

    def _map_access(self, expr, name, index):
        if name in self.kernel.all_inames():
            return set()

        descr = self.kernel.get_var_descriptor(name)
        if descr.address_space == self.address_space:
            result = set([expr])
        else:
            result = set()

        return result | self.rec(index)

    def map_variable(self, expr):
        return self._map_access(expr, expr.name, ())

    def map_subscript(self, expr):
        import pymbolic.primitives as p
        assert isinstance(expr.aggregate, p.Variable)
        return self._map_access(expr, expr.aggregate.name, expr.index)


def _make_grid_size_domain(kernel, var_name_gen=None):
    if var_name_gen is None:
        var_name_gen = kernel.get_var_name_generator()

    ggrid, lgrid = kernel.get_grid_size_upper_bounds()
    ggrid_var_names = [var_name_gen("gid%d" % axis) for axis in range(len(ggrid))]
    lgrid_var_names = [var_name_gen("lid%d" % axis) for axis in range(len(lgrid))]
    grid_var_pwaffs = isl.make_zero_and_vars(
            ggrid_var_names + lgrid_var_names, kernel.all_params())

    grid_range_dom = grid_var_pwaffs[0].le_set(grid_var_pwaffs[0])
    for var, ubound in zip(ggrid_var_names + lgrid_var_names, ggrid + lgrid):
        ubound = isl.align_spaces(ubound, grid_var_pwaffs[0])
        grid_range_dom = grid_range_dom & (
                grid_var_pwaffs[0].le_set(grid_var_pwaffs[var])
                &
                grid_var_pwaffs[var].lt_set(ubound))

    grid_range_dom, = grid_range_dom.get_basic_sets()

    return ggrid_var_names, lgrid_var_names, grid_range_dom


def remove_work(kernel):
    """This transform removes operations in a kernel, leaving only
    accesses to global memory.

    .. note::

        This routine will currently not work correctly in the presence of
        data-dependent flow control or memory access.
    """
    import loopy as lp
    import pymbolic.primitives as p

    kernel = lp.preprocess_kernel(kernel)

    gatherer = _MemAccessGatherer(kernel, lp.AddressSpace.GLOBAL)

    from loopy.kernel.instruction import MultiAssignmentBase, make_assignment

    # maps each old ID to a frozenset of new IDs
    old_to_new_ids = {}
    insn_id_gen = kernel.get_instruction_id_generator()

    var_name_gen = kernel.get_var_name_generator()
    read_tgt_var_name = var_name_gen("read_tgt")
    new_temporary_variables = kernel.temporary_variables.copy()
    new_temporary_variables[read_tgt_var_name] = lp.TemporaryVariable(
            read_tgt_var_name, address_space=lp.AddressSpace.PRIVATE)

    new_instructions = []

    # {{{ create init insn for read target

    ggrid_var_names, lgrid_var_names, grid_range_dom = _make_grid_size_domain(kernel)
    grid_inames = frozenset(ggrid_var_names + lgrid_var_names)

    read_tgt_init_id = insn_id_gen("init_read_tgt")
    old_to_new_ids[read_tgt_init_id] = [read_tgt_init_id]
    new_instructions.append(
            make_assignment(
                (p.Variable(read_tgt_var_name),),
                0,
                id=read_tgt_init_id,
                within_inames=grid_inames))

    # }}}

    # {{{ rewrite instructions

    read_insn_ids = []

    for insn in kernel.instructions:
        if not isinstance(insn, MultiAssignmentBase):
            new_instructions.append(insn)
            old_to_new_ids[insn.id] = frozenset([insn.id])
            continue

        writer_accesses = set.union(*[
            gatherer(lhs) for lhs in insn.assignees])

        reader_accesses = gatherer(insn.expression)

        new_insn_ids = set()
        for read_expr in reader_accesses:
            new_id = insn_id_gen(insn.id)
            read_insn_ids.append(insn.id)
            new_instructions.append(
                    make_assignment(
                        (p.Variable(read_tgt_var_name),),
                        p.Variable(read_tgt_var_name) + read_expr,
                        id=new_id,
                        within_inames=insn.within_inames,
                        depends_on=insn.depends_on | frozenset([read_tgt_init_id])))
            new_insn_ids.add(new_id)

        for write_expr in writer_accesses:
            new_id = insn_id_gen(insn.id)
            new_instructions.append(
                    make_assignment(
                        (write_expr,),
                        17,
                        id=new_id,
                        within_inames=insn.within_inames,
                        depends_on=insn.depends_on))
            new_insn_ids.add(new_id)

        old_to_new_ids[insn.id] = frozenset(new_insn_ids)

    # }}}

    # {{{ create write-out insn for read target

    _, lgrid = kernel.get_grid_size_upper_bounds_as_exprs()
    read_tgt_local_dest_name = var_name_gen("read_tgt_dest")
    new_temporary_variables[read_tgt_local_dest_name] = lp.TemporaryVariable(
            name=read_tgt_local_dest_name,
            address_space=lp.AddressSpace.LOCAL,
            shape=lgrid)

    write_read_tgt_id = insn_id_gen("write_read_tgt")
    old_to_new_ids[write_read_tgt_id] = [write_read_tgt_id]
    new_instructions.append(
        make_assignment(
            (p.Variable(read_tgt_local_dest_name)[
                tuple(p.Variable(lgn) for lgn in lgrid_var_names)],),
            p.Variable(read_tgt_var_name),
            id=write_read_tgt_id,
            depends_on=frozenset(read_insn_ids),
            within_inames=grid_inames))

    # }}}

    # {{{ rewrite dependencies for new IDs

    new_instructions_2 = []

    for insn in new_instructions:
        new_instructions_2.append(
                insn.copy(
                    depends_on=frozenset(
                        subdep
                        for dep in insn.depends_on
                        for subdep in old_to_new_ids[dep])))

    # }}}

    kernel = kernel.copy(
            domains=kernel.domains + [grid_range_dom],
            state=lp.KernelState.INITIAL,
            instructions=new_instructions_2,
            temporary_variables=new_temporary_variables)

    from loopy.kernel.data import GroupIndexTag, LocalIndexTag
    kernel = lp.tag_inames(kernel, dict(
        (ggrid_var_names[i], GroupIndexTag(i))
        for i in range(len(ggrid_var_names))))
    kernel = lp.tag_inames(kernel, dict(
        (lgrid_var_names[i], LocalIndexTag(i))
        for i in range(len(lgrid_var_names))))

    return kernel

# }}}

# vim: foldmethod=marker

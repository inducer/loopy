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
from loopy.kernel import LoopKernel
from loopy.kernel.function_interface import (ScalarCallable, CallableKernel)
from loopy.translation_unit import (TranslationUnit,
                                    for_each_kernel)


# {{{ find_instructions

def find_instructions_in_single_kernel(kernel, insn_match):
    assert isinstance(kernel, LoopKernel)
    from loopy.match import parse_match
    match = parse_match(insn_match)
    return [insn for insn in kernel.instructions if match(kernel, insn)]


def find_instructions(program, insn_match):
    if isinstance(program, LoopKernel):
        return find_instructions_in_single_kernel(program, insn_match)

    assert isinstance(program, TranslationUnit)
    insns = []
    for in_knl_callable in program.callables_table.values():
        if isinstance(in_knl_callable, CallableKernel):
            insns += (find_instructions_in_single_kernel(
                in_knl_callable.subkernel, insn_match))
        elif isinstance(in_knl_callable, ScalarCallable):
            pass
        else:
            raise NotImplementedError("Unknown callable type %s." % (
                type(in_knl_callable)))

    return insns

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

@for_each_kernel
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

@for_each_kernel
def add_dependency(kernel, insn_match, depends_on):
    """Add dependency contained in *depends_on* to the instructions matched
    by *insn_match*.

    :arg kernel: A :class:`loopy.LoopKernel`.

    :arg insn_match: An instruction id match understood by
        :func:`loopy.match.parse_match` identifying the statement to which the
        dependency will be added.

    :arg depends_on: A two-tuple containing an instruction id match understood by
        :func:`loopy.match.parse_match` identifying the dependee statement(s),
        and an class:`islpy.Map` from each instance of the dependee
        statement(s) to all instances of the depender statement(s) that must occur
        later. For backward compatability, *depends_on* may also be an
        instruction id match identifying dependee statement ids to be added to
        `stmt.depends_on` for any stmt matching *insn_match*.

    :returns: The updated :class:`loopy.LoopKernel` with the new dependency.

    .. versionchanged:: 2016.3

        Third argument renamed to *depends_on* for clarity, allowed to
        be not just ID but also match expression.
    """

    # Determine whether we received legacy or contemporary dependency
    if isinstance(depends_on, tuple):
        dep_map = depends_on[1]
        depends_on = depends_on[0]
    else:
        dep_map = None

    if isinstance(depends_on, str) and depends_on in kernel.id_to_insn:
        dependee_ids = frozenset([depends_on])
    else:
        dependee_ids = frozenset(
                dependee.id for dependee in find_instructions_in_single_kernel(
                    kernel, depends_on))

    if not dependee_ids:
        raise LoopyError("no instructions found matching '%s' "
                "(to add as dependencies)" % depends_on)

    matched = [False]

    if dep_map is None:
        # Handle legacy dependencies
        def add_dep(insn):
            new_deps = insn.depends_on  # Set of dependee ids
            matched[0] = True
            if new_deps is None:
                new_deps = dependee_ids
            else:
                new_deps = new_deps | dependee_ids
            return insn.copy(depends_on=new_deps)
    else:
        # Handle contemporary dependencies
        def add_dep(stmt):
            new_deps_dict = stmt.dependencies  # Mapping of dependee ids to dep maps
            matched[0] = True
            for dependee_id in dependee_ids:
                new_deps_dict.setdefault(dependee_id, []).append(dep_map)
            return stmt.copy(dependencies=new_deps_dict)

    result = map_instructions(kernel, insn_match, add_dep)

    if not matched[0]:
        raise LoopyError("no instructions found matching '%s' "
                "(to which dependencies would be added)" % insn_match)

    return result

# }}}


# {{{ remove_instructions

def _toposort_of_subset_of_insns(kernel, subset_insns):
    """
    Returns a :class:`list` of insn ids which is a topological sort of insn
    deps in *subset_insns*.

    :arg subset_insns: a :class:`frozenset` of insn ids that are a subset of
        kernel over which we wish to compute the topological sort.
    """
    dag = {insn_id: set(kernel.id_to_insn[insn_id].depends_on
                        & subset_insns)
           for insn_id in subset_insns}

    from pytools.graph import compute_topological_order

    return compute_topological_order(dag)[::-1]


@for_each_kernel
def remove_instructions(kernel, insn_ids):
    """Return a new kernel with instructions in *insn_ids* removed.

    Dependencies across deleted instructions are transitively propagated i.e.
    if insn_a depends on insn_b that depends on insn_c and  'insn_b' is to be
    removed then the returned kernel will have a dependency from 'insn_a' to
    'insn_c'.

    This also updates *no_sync_with* for all instructions.

    :arg insn_ids: An instance of :class:`set` or :class:`str` as
        understood by :func:`loopy.match.parse_match` or
        :class:`loopy.match.MatchExpressionBase`.
    """
    from functools import reduce

    if not insn_ids:
        return kernel

    from loopy.match import MatchExpressionBase

    if isinstance(insn_ids, str):
        from loopy.match import parse_match
        insn_ids = parse_match(insn_ids)
    if isinstance(insn_ids, MatchExpressionBase):
        within = insn_ids

        insn_ids = set([insn.id for insn in kernel.instructions if
            within(kernel, insn)])

    assert isinstance(insn_ids, set)
    id_to_insn = kernel.id_to_insn

    # {{{ for each insn_id to be removed get deps in terms of remaining insns

    # transitive_deps: mapping from insn_id (referred as I) to be removed to
    # frozenset of insn_ids that won't be removed (referred as R(I)). 'R(I)' are
    # the transitive dependencies of 'I' that won't be removed.

    transitive_deps = {}
    insns_not_to_be_removed = frozenset(id_to_insn) - insn_ids

    for insn_id in _toposort_of_subset_of_insns(kernel, insn_ids):
        assert id_to_insn[insn_id].depends_on <= (insns_not_to_be_removed
                                                  | frozenset(transitive_deps))
        transitive_deps[insn_id] = reduce(frozenset.union,
                                          (transitive_deps.get(d, frozenset([d]))
                                           for d in id_to_insn[insn_id].depends_on),
                                          frozenset())

    # }}}

    new_insns = []
    for insn in kernel.instructions:
        if insn.id in insn_ids:
            continue

        # transitively propagate dependencies
        if insn.depends_on is None:
            depends_on = frozenset()
        else:
            depends_on = insn.depends_on

        new_deps = reduce(frozenset.union,
                          (transitive_deps.get(d, frozenset([d]))
                           for d in depends_on),
                          frozenset())

        assert (new_deps & insn_ids) == frozenset()

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
    if not replacements:
        return kernel

    new_insns = []

    for insn in kernel.instructions:
        changed = False
        new_depends_on = list(insn.depends_on)
        extra_depends_on = []
        new_no_sync_with = []

        for idep, dep in enumerate(insn.depends_on):
            if dep in replacements:
                new_deps = list(replacements[dep])
                new_depends_on[idep] = new_deps[0]
                extra_depends_on.extend(new_deps[1:])
                changed = True

        for insn_id, scope in insn.no_sync_with:
            if insn_id in replacements:
                new_no_sync_with.extend(
                        (repl, scope) for repl in replacements[insn_id])
                changed = True
            else:
                new_no_sync_with.append((insn_id, scope))

        new_insns.append(
                insn.copy(
                    depends_on=frozenset(new_depends_on + extra_depends_on),
                    no_sync_with=frozenset(new_no_sync_with))
                if changed else insn)

    return kernel.copy(instructions=new_insns)

# }}}


# {{{ tag_instructions

@for_each_kernel
def tag_instructions(kernel, new_tag, within=None):
    from loopy.match import parse_match
    within = parse_match(within)

    from loopy.kernel.creation import _normalize_tags
    new_tags = _normalize_tags([new_tag])

    new_insns = []
    for insn in kernel.instructions:
        if within(kernel, insn):
            new_insns.append(
                    insn.copy(tags=insn.tags | new_tags))
        else:
            new_insns.append(insn)

    return kernel.copy(instructions=new_insns)

# }}}


# {{{ add nosync

@for_each_kernel
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
    assert isinstance(kernel, LoopKernel)

    if isinstance(source, str) and source in kernel.id_to_insn:
        sources = frozenset([source])
    else:
        sources = frozenset(
                source.id for source in find_instructions_in_single_kernel(
                    kernel, source))

    if isinstance(sink, str) and sink in kernel.id_to_insn:
        sinks = frozenset([sink])
    else:
        sinks = frozenset(
                sink.id for sink in find_instructions_in_single_kernel(
                    kernel, sink))

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

@for_each_kernel
def uniquify_instruction_ids(kernel):
    """Converts any ids that are :class:`loopy.UniqueName` or *None* into unique
    strings.

    This function does *not* deduplicate existing instruction ids.
    """

    from loopy.kernel.creation import UniqueName

    insn_ids = {
            insn.id for insn in kernel.instructions
            if insn.id is not None and not isinstance(insn.id, UniqueName)}

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


# vim: foldmethod=marker

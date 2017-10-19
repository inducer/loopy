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

from loopy.diagnostic import LoopyError


# {{{ find_statements

def find_statements(kernel, stmt_match):
    from loopy.match import parse_match
    match = parse_match(stmt_match)
    return [stmt for stmt in kernel.statements if match(kernel, stmt)]

# }}}


# {{{ map_statements

def map_statements(kernel, stmt_match, f):
    from loopy.match import parse_match
    match = parse_match(stmt_match)

    new_stmts = []

    for stmt in kernel.statements:
        if match(kernel, stmt):
            new_stmts.append(f(stmt))
        else:
            new_stmts.append(stmt)

    return kernel.copy(statements=new_stmts)

# }}}


# {{{ set_statement_priority

def set_statement_priority(kernel, stmt_match, priority):
    """Set the priority of statements matching *stmt_match* to *priority*.

    *stmt_match* may be any statement id match understood by
    :func:`loopy.match.parse_match`.
    """

    def set_prio(stmt):
        return stmt.copy(priority=priority)

    return map_statements(kernel, stmt_match, set_prio)

# }}}


# {{{ add_dependency

def add_dependency(kernel, stmt_match, depends_on):
    """Add the statement dependency *dependency* to the statements matched
    by *stmt_match*.

    *stmt_match* and *depends_on* may be any statement id match understood by
    :func:`loopy.match.parse_match`.

    .. versionchanged:: 2016.3

        Third argument renamed to *depends_on* for clarity, allowed to
        be not just ID but also match expression.
    """

    if isinstance(depends_on, str) and depends_on in kernel.id_to_stmt:
        added_deps = frozenset([depends_on])
    else:
        added_deps = frozenset(
                dep.id for dep in find_statements(kernel, depends_on))

    if not added_deps:
        raise LoopyError("no statements found matching '%s' "
                "(to add as dependencies)" % depends_on)

    matched = [False]

    def add_dep(stmt):
        new_deps = stmt.depends_on
        matched[0] = True
        if new_deps is None:
            new_deps = added_deps
        else:
            new_deps = new_deps | added_deps

        return stmt.copy(depends_on=new_deps)

    result = map_statements(kernel, stmt_match, add_dep)

    if not matched[0]:
        raise LoopyError("no statements found matching '%s' "
                "(to which dependencies would be added)" % stmt_match)

    return result

# }}}


# {{{ remove_statements

def remove_statements(kernel, stmt_ids):
    """Return a new kernel with statements in *stmt_ids* removed.

    Dependencies across (one, for now) deleted isntructions are propagated.
    Behavior is undefined for now for chains of dependencies within the
    set of deleted statements.

    This also updates *no_sync_with* for all statements.
    """

    if not stmt_ids:
        return kernel

    assert isinstance(stmt_ids, set)
    id_to_stmt = kernel.id_to_stmt

    new_stmts = []
    for stmt in kernel.statements:
        if stmt.id in stmt_ids:
            continue

        # transitively propagate dependencies
        # (only one level for now)
        if stmt.depends_on is None:
            depends_on = frozenset()
        else:
            depends_on = stmt.depends_on

        new_deps = depends_on - stmt_ids

        for dep_id in depends_on & stmt_ids:
            new_deps = new_deps | id_to_stmt[dep_id].depends_on

        # update no_sync_with

        new_no_sync_with = frozenset((stmt_id, scope)
                for stmt_id, scope in stmt.no_sync_with
                if stmt_id not in stmt_ids)

        new_stmts.append(
                stmt.copy(depends_on=new_deps, no_sync_with=new_no_sync_with))

    return kernel.copy(
            statements=new_stmts)

# }}}


# {{{ replace_statement_ids

def replace_statement_ids(kernel, replacements):
    new_stmts = []

    for stmt in kernel.statements:
        changed = False
        new_depends_on = []
        new_no_sync_with = []

        for dep in stmt.depends_on:
            if dep in replacements:
                new_depends_on.extend(replacements[dep])
                changed = True
            else:
                new_depends_on.append(dep)

        for stmt_id, scope in stmt.no_sync_with:
            if stmt_id in replacements:
                new_no_sync_with.extend(
                        (repl, scope) for repl in replacements[stmt_id])
                changed = True
            else:
                new_no_sync_with.append((stmt_id, scope))

        new_stmts.append(
                stmt.copy(
                    depends_on=frozenset(new_depends_on),
                    no_sync_with=frozenset(new_no_sync_with))
                if changed else stmt)

    return kernel.copy(statements=new_stmts)

# }}}


# {{{ tag_statements

def tag_statements(kernel, new_tag, within=None):
    from loopy.match import parse_match
    within = parse_match(within)

    new_stmts = []
    for stmt in kernel.statements:
        if within(kernel, stmt):
            new_stmts.append(
                    stmt.copy(tags=stmt.tags | frozenset([new_tag])))
        else:
            new_stmts.append(stmt)

    return kernel.copy(statements=new_stmts)

# }}}


# {{{ add nosync

def add_nosync(kernel, scope, source, sink, bidirectional=False, force=False):
    """Add a *no_sync_with* directive between *source* and *sink*.
    *no_sync_with* is only added if *sink* depends on *source* or
    if the statement pair is in a conflicting group.

    This function does not check for the presence of a memory dependency.

    :arg kernel: The kernel
    :arg source: Either a single statement id, or any statement id
        match understood by :func:`loopy.match.parse_match`.
    :arg sink: Either a single statement id, or any statement id
        match understood by :func:`loopy.match.parse_match`.
    :arg scope: A valid *no_sync_with* scope. See
        :attr:`loopy.StatementBase.no_sync_with` for allowable scopes.
    :arg bidirectional: A :class:`bool`. If *True*, add a *no_sync_with*
        to both the source and sink statements, otherwise the directive
        is only added to the sink statements.
    :arg force: A :class:`bool`. If *True*, add a *no_sync_with* directive
        even without the presence of a dependency edge or conflicting
        statement group.

    :return: The updated kernel
    """

    if isinstance(source, str) and source in kernel.id_to_stmt:
        sources = frozenset([source])
    else:
        sources = frozenset(
                source.id for source in find_statements(kernel, source))

    if isinstance(sink, str) and sink in kernel.id_to_stmt:
        sinks = frozenset([sink])
    else:
        sinks = frozenset(
                sink.id for sink in find_statements(kernel, sink))

    def stmts_in_conflicting_groups(stmt1_id, stmt2_id):
        stmt1 = kernel.id_to_stmt[stmt1_id]
        stmt2 = kernel.id_to_stmt[stmt2_id]
        return (
                bool(stmt1.groups & stmt2.conflicts_with_groups)
                or
                bool(stmt2.groups & stmt1.conflicts_with_groups))

    from collections import defaultdict
    nosync_to_add = defaultdict(set)

    for sink in sinks:
        for source in sources:

            needs_nosync = force or (
                    source in kernel.recursive_stmt_dep_map()[sink]
                    or stmts_in_conflicting_groups(source, sink))

            if not needs_nosync:
                continue

            nosync_to_add[sink].add((source, scope))
            if bidirectional:
                nosync_to_add[source].add((sink, scope))

    new_statements = list(kernel.statements)

    for i, stmt in enumerate(new_statements):
        if stmt.id in nosync_to_add:
            new_statements[i] = stmt.copy(no_sync_with=stmt.no_sync_with
                    | frozenset(nosync_to_add[stmt.id]))

    return kernel.copy(statements=new_statements)

# }}}


# {{{ uniquify_statement_ids

def uniquify_statement_ids(kernel):
    """Converts any ids that are :class:`loopy.UniqueName` or *None* into unique
    strings.

    This function does *not* deduplicate existing statement ids.
    """

    from loopy.kernel.creation import UniqueName

    stmt_ids = set(
            stmt.id for stmt in kernel.statements
            if stmt.id is not None and not isinstance(stmt.id, UniqueName))

    from pytools import UniqueNameGenerator
    stmt_id_gen = UniqueNameGenerator(stmt_ids)

    new_statements = []

    for stmt in kernel.statements:
        if stmt.id is None:
            new_statements.append(
                    stmt.copy(id=stmt_id_gen("stmt")))
        elif isinstance(stmt.id, UniqueName):
            new_statements.append(
                    stmt.copy(id=stmt_id_gen(stmt.id.name)))
        else:
            new_statements.append(stmt)

    return kernel.copy(statements=new_statements)

# }}}


# vim: foldmethod=marker

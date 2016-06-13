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

def add_dependency(kernel, insn_match, dependency):
    """Add the instruction dependency *dependency* to the instructions matched
    by *insn_match*.

    *insn_match* may be any instruction id match understood by
    :func:`loopy.match.parse_match`.
    """

    if dependency not in kernel.id_to_insn:
        raise LoopyError("cannot add dependency on non-existent instruction ID '%s'"
                % dependency)

    def add_dep(insn):
        new_deps = insn.depends_on
        added_deps = frozenset([dependency])
        if new_deps is None:
            new_deps = added_deps
        else:
            new_deps = new_deps | added_deps

        return insn.copy(depends_on=new_deps)

    return map_instructions(kernel, insn_match, add_dep)

# }}}


# {{{ remove_instructions

def remove_instructions(kernel, insn_ids):
    """Return a new kernel with instructions in *insn_ids* removed.

    Dependencies across (one, for now) deleted isntructions are propagated.
    Behavior is undefined for now for chains of dependencies within the
    set of deleted instructions.
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

        new_insns.append(insn.copy(depends_on=frozenset(new_deps)))

    return kernel.copy(
            instructions=new_insns)

# }}}


# {{{ replace_instruction_ids

def replace_instruction_ids(kernel, replacements):
    new_insns = []

    for insn in kernel.instructions:
        changed = False
        new_depends_on = []

        for dep in insn.depends_on:
            if dep in replacements:
                new_depends_on.extend(replacements[dep])
                changed = True
            else:
                new_depends_on.append(dep)

        new_insns.append(
                insn.copy(depends_on=frozenset(new_depends_on))
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


# vim: foldmethod=marker

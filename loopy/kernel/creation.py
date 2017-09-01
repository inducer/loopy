"""UI for kernel creation."""

from __future__ import division, absolute_import, print_function

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


import numpy as np

from pymbolic.mapper import CSECachingMapperMixin
from loopy.tools import intern_frozenset_of_ids
from loopy.symbolic import IdentityMapper, WalkMapper
from loopy.kernel.data import (
        InstructionBase,
        MultiAssignmentBase, Assignment,
        SubstitutionRule)
from loopy.diagnostic import LoopyError, warn_with_kernel
import islpy as isl
from islpy import dim_type

import six
from six.moves import range, zip, intern

import re

import logging
logger = logging.getLogger(__name__)


# {{{ identifier wrangling

_IDENTIFIER_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")


def _gather_isl_identifiers(s):
    return set(_IDENTIFIER_RE.findall(s)) - set(["and", "or", "exists"])


class UniqueName:
    """A tag for a string that identifies a partial identifier that is to
    be made unique by the UI.
    """

    def __init__(self, name):
        self.name = name

# }}}


# {{{ expand defines

WORD_RE = re.compile(r"\b([a-zA-Z0-9_]+)\b")
BRACE_RE = re.compile(r"\$\{([a-zA-Z0-9_]+)\}")


def expand_defines(insn, defines, single_valued=True):
    replacements = [()]

    processed_defines = set()

    for find_regexp, replace_pattern in [
            (BRACE_RE, r"\$\{%s\}"),
            (WORD_RE, r"\b%s\b"),
            ]:

        for match in find_regexp.finditer(insn):
            define_name = match.group(1)

            # {{{ don't process the same define multiple times

            if define_name in processed_defines:
                # already dealt with
                continue

            processed_defines.add(define_name)

            # }}}

            try:
                value = defines[define_name]
            except KeyError:
                continue

            if isinstance(value, list):
                if single_valued:
                    raise ValueError("multi-valued macro expansion "
                            "not allowed "
                            "in this context (when expanding '%s')" % define_name)

                replacements = [
                        rep+((replace_pattern % define_name, subval),)
                        for rep in replacements
                        for subval in value
                        ]
            else:
                replacements = [
                        rep+((replace_pattern % define_name, value),)
                        for rep in replacements]

    for rep in replacements:
        rep_value = insn
        for pattern, val in rep:
            rep_value = re.sub(pattern, str(val), rep_value)

        yield rep_value


def expand_defines_in_expr(expr, defines):
    from pymbolic.primitives import Variable
    from loopy.symbolic import parse

    def subst_func(var):
        if isinstance(var, Variable):
            try:
                var_value = defines[var.name]
            except KeyError:
                return None
            else:
                return parse(str(var_value))
        else:
            return None

    from loopy.symbolic import SubstitutionMapper, PartialEvaluationMapper
    return PartialEvaluationMapper()(
            SubstitutionMapper(subst_func)(expr))

# }}}


# {{{ instruction options

def get_default_insn_options_dict():
    return {
        "depends_on": frozenset(),
        "depends_on_is_final": False,
        "no_sync_with": frozenset(),
        "groups": frozenset(),
        "conflicts_with_groups": frozenset(),
        "insn_id": None,
        "inames_to_dup": [],
        "priority": 0,
        "within_inames_is_final": False,
        "within_inames": frozenset(),
        "predicates": frozenset(),
        "tags": frozenset(),
        "atomicity": (),
        }


from collections import namedtuple

_NosyncParseResult = namedtuple("_NosyncParseResult", "expr, scope")


def parse_insn_options(opt_dict, options_str, assignee_names=None):
    if options_str is None:
        return opt_dict

    is_with_block = assignee_names is None

    result = opt_dict.copy()

    def parse_nosync_option(opt_value):
        if "@" in opt_value:
            expr, scope = opt_value.split("@")
        else:
            expr = opt_value
            scope = "any"
        allowable_scopes = ("local", "global", "any")
        if scope not in allowable_scopes:
            raise ValueError(
                "unknown scope for nosync option: '%s' "
                "(allowable scopes are %s)" %
                (scope, ', '.join("'%s'" % s for s in allowable_scopes)))
        return _NosyncParseResult(expr, scope)

    for option in options_str.split(","):
        option = option.strip()
        if not option:
            raise RuntimeError("empty option supplied")

        equal_idx = option.find("=")
        if equal_idx == -1:
            opt_key = option
            opt_value = None
        else:
            opt_key = option[:equal_idx].strip()
            opt_value = option[equal_idx+1:].strip()

        if opt_key == "id" and opt_value is not None:
            if is_with_block:
                raise LoopyError("'id' option may not be specified "
                        "in a 'with' block")

            result["insn_id"] = intern(opt_value)

        elif opt_key == "id_prefix" and opt_value is not None:
            result["insn_id"] = UniqueName(opt_value)

        elif opt_key == "priority" and opt_value is not None:
            if is_with_block:
                raise LoopyError("'priority' option may not be specified "
                        "in a 'with' block")

            result["priority"] = int(opt_value)

        elif opt_key == "dup" and opt_value is not None:
            if is_with_block:
                raise LoopyError("'dup' option may not be specified "
                        "in a 'with' block")

            for value in opt_value.split(":"):
                arrow_idx = value.find("->")
                if arrow_idx >= 0:
                    result["inames_to_dup"] = (
                            result.get("inames_to_dup", [])
                            +
                            [(value[:arrow_idx], value[arrow_idx+2:])])
                else:
                    result["inames_to_dup"] = (
                            result.get("inames_to_dup", [])
                            +
                            [(value, None)])

        elif opt_key == "dep" and opt_value is not None:
            if opt_value.startswith("*"):
                result["depends_on_is_final"] = True
                opt_value = (opt_value[1:]).strip()

            result["depends_on"] = result["depends_on"].union(frozenset(
                    intern(dep.strip()) for dep in opt_value.split(":")
                    if dep.strip()))

        elif opt_key == "dep_query" and opt_value is not None:
            from loopy.match import parse_match
            match = parse_match(opt_value)
            result["depends_on"] = result["depends_on"].union(frozenset([match]))

        elif opt_key == "nosync" and opt_value is not None:
            if is_with_block:
                raise LoopyError("'nosync' option may not be specified "
                        "in a 'with' block")

            result["no_sync_with"] = result["no_sync_with"].union(frozenset(
                    (option.expr.strip(), option.scope)
                    for option in (
                            parse_nosync_option(entry)
                            for entry in opt_value.split(":"))
                    if option.expr.strip()))

        elif opt_key == "nosync_query" and opt_value is not None:
            if is_with_block:
                raise LoopyError("'nosync' option may not be specified "
                        "in a 'with' block")

            match_expr, scope = parse_nosync_option(opt_value)

            from loopy.match import parse_match
            match = parse_match(match_expr)
            result["no_sync_with"] = result["no_sync_with"].union(
                    frozenset([(match, scope)]))

        elif opt_key == "groups" and opt_value is not None:
            result["groups"] = frozenset(
                    intern(grp.strip()) for grp in opt_value.split(":")
                    if grp.strip())

        elif opt_key == "conflicts" and opt_value is not None:
            result["conflicts_with_groups"] = frozenset(
                    intern(grp.strip()) for grp in opt_value.split(":")
                    if grp.strip())

        elif opt_key == "inames" and opt_value is not None:
            if opt_value.startswith("+"):
                result["within_inames_is_final"] = False
                opt_value = (opt_value[1:]).strip()
            else:
                result["within_inames_is_final"] = True

            result["within_inames"] = intern_frozenset_of_ids(
                    opt_value.split(":"))

        elif opt_key == "if" and opt_value is not None:
            predicates = opt_value.split(":")
            new_predicates = set()

            for pred in predicates:
                from pymbolic.primitives import LogicalNot
                from loopy.symbolic import parse
                if pred.startswith("!"):
                    from warnings import warn
                    warn("predicates starting with '!' are deprecated. "
                            "Simply use 'not' instead")
                    pred = LogicalNot(parse(pred[1:]))
                else:
                    pred = parse(pred)

                new_predicates.add(pred)

            result["predicates"] = frozenset(new_predicates)

            del predicates
            del new_predicates

        elif opt_key == "tags" and opt_value is not None:
            result["tags"] = frozenset(
                    tag.strip() for tag in opt_value.split(":")
                    if tag.strip())

        elif opt_key == "atomic":
            if is_with_block:
                raise LoopyError("'atomic' option may not be specified "
                        "in a with block")

            if len(assignee_names) != 1:
                raise LoopyError("atomic operations with more than one "
                        "left-hand side not supported")
            assignee_name, = assignee_names

            import loopy as lp
            if opt_value is None:
                result["atomicity"] = result["atomicity"] + (
                        lp.AtomicUpdate(assignee_name),)
            else:
                for v in opt_value.split(":"):
                    if v == "init":
                        result["atomicity"] = result["atomicity"] + (
                                lp.AtomicInit(assignee_name),)
                    else:
                        raise LoopyError("atomicity directive not "
                                "understood: %s"
                                % v)
            del assignee_name

        else:
            raise ValueError(
                    "unrecognized instruction option '%s' "
                    "(maybe a missing/extraneous =value?)"
                    % opt_key)

    return result

# }}}


# {{{ parse one instruction

WITH_OPTIONS_RE = re.compile(
        r"^"
        r"\s*with\s*"
        r"\{(?P<options>.+)\}"
        r"\s*$")

FOR_RE = re.compile(
        r"^"
        r"\s*(for)\s+"
        r"(?P<inames>[ ,\w]*)"
        r"\s*$")

IF_RE = re.compile(
        r"^"
        r"\s*if\s+"
        r"(?P<predicate>.+)"
        r"\s*$")

ELIF_RE = re.compile(
        r"^"
        r"\s*elif\s+"
        r"(?P<predicate>.+)"
        r"\s*$")

ELSE_RE = re.compile(r"^\s*else\s*$")

INSN_RE = re.compile(
        r"^"
        r"\s*"
        r"(?P<lhs>[^{]+?)"
        r"\s*(?<!\:)=\s*"
        r"(?P<rhs>.+?)"
        r"\s*?"
        r"(?:\{(?P<options>.+)\}\s*)?$")

EMPTY_LHS_INSN_RE = re.compile(
        r"^"
        r"\s*"
        r"(?P<rhs>.+?)"
        r"\s*?"
        r"(?:\{(?P<options>.+)\}\s*)?$")

SPECIAL_INSN_RE = re.compile(
        r"^"
        r"\s*"
        r"\.\.\."
        r"\s*"
        r"(?P<kind>[a-z]+?)"
        r"\s*?"
        r"(?:\{(?P<options>.+)\}\s*)?$")

SUBST_RE = re.compile(
        r"^\s*(?P<lhs>.+?)\s*:=\s*(?P<rhs>.+)\s*$")


def parse_insn(groups, insn_options):
    """
    :return: a tuple ``(insn, inames_to_dup)``, where insn is a
        :class:`Assignment`, a :class:`CallInstruction`,
        or a :class:`SubstitutionRule`
        and *inames_to_dup* is None or a list of tuples `(old, new)`.
    """

    import loopy as lp
    from loopy.symbolic import parse

    if "lhs" in groups:
        try:
            lhs = parse(groups["lhs"])
        except:
            print("While parsing left hand side '%s', "
                    "the following error occurred:" % groups["lhs"])
            raise
    else:
        lhs = ()

    try:
        rhs = parse(groups["rhs"])
    except:
        print("While parsing right hand side '%s', "
                "the following error occurred:" % groups["rhs"])
        raise

    from pymbolic.primitives import Variable, Subscript, Lookup
    from loopy.symbolic import TypeAnnotation

    if not isinstance(lhs, tuple):
        lhs = (lhs,)

    temp_var_types = []
    new_lhs = []
    assignee_names = []

    for lhs_i in lhs:
        if isinstance(lhs_i, TypeAnnotation):
            if lhs_i.type is None:
                temp_var_types.append(lp.auto)
            else:
                temp_var_types.append(lhs_i.type)

            lhs_i = lhs_i.child
        else:
            temp_var_types.append(None)

        inner_lhs_i = lhs_i
        if isinstance(inner_lhs_i, Lookup):
            inner_lhs_i = inner_lhs_i.aggregate

        from loopy.symbolic import LinearSubscript
        if isinstance(inner_lhs_i, Variable):
            assignee_names.append(inner_lhs_i.name)
        elif isinstance(inner_lhs_i, (Subscript, LinearSubscript)):
            assignee_names.append(inner_lhs_i.aggregate.name)
        else:
            raise LoopyError("left hand side of assignment '%s' must "
                    "be variable or subscript" % (lhs_i,))

        new_lhs.append(lhs_i)

    lhs = tuple(new_lhs)
    temp_var_types = tuple(temp_var_types)
    del new_lhs

    insn_options = parse_insn_options(
            insn_options.copy(),
            groups["options"],
            assignee_names=assignee_names)

    insn_id = insn_options.pop("insn_id", None)
    inames_to_dup = insn_options.pop("inames_to_dup", [])

    kwargs = dict(
                id=(
                    intern(insn_id)
                    if isinstance(insn_id, str)
                    else insn_id),
                **insn_options)

    from loopy.kernel.instruction import make_assignment
    return make_assignment(
            lhs, rhs, temp_var_types, **kwargs
            ), inames_to_dup

# }}}


# {{{ parse_subst_rule

def parse_subst_rule(groups):
    from loopy.symbolic import parse
    try:
        lhs = parse(groups["lhs"])
    except:
        print("While parsing left hand side '%s', "
                "the following error occurred:" % groups["lhs"])
        raise

    try:
        rhs = parse(groups["rhs"])
    except:
        print("While parsing right hand side '%s', "
                "the following error occurred:" % groups["rhs"])
        raise

    from pymbolic.primitives import Variable, Call
    if isinstance(lhs, Variable):
        subst_name = lhs.name
        arg_names = []
    elif isinstance(lhs, Call):
        if not isinstance(lhs.function, Variable):
            raise RuntimeError("Invalid substitution rule left-hand side")
        subst_name = lhs.function.name
        arg_names = []

        for i, arg in enumerate(lhs.parameters):
            if not isinstance(arg, Variable):
                raise RuntimeError("Invalid substitution rule "
                                "left-hand side: %s--arg number %d "
                                "is not a variable" % (lhs, i))
            arg_names.append(arg.name)
    else:
        raise RuntimeError("Invalid substitution rule left-hand side")

    return SubstitutionRule(
            name=subst_name,
            arguments=tuple(arg_names),
            expression=rhs)

# }}}


# {{{ parse_special_insn

def parse_special_insn(groups, insn_options):
    insn_options = parse_insn_options(
            insn_options.copy(),
            groups["options"],
            assignee_names=())

    del insn_options["atomicity"]

    insn_id = insn_options.pop("insn_id", None)
    inames_to_dup = insn_options.pop("inames_to_dup", [])

    kwargs = dict(
                id=(
                    intern(insn_id)
                    if isinstance(insn_id, str)
                    else insn_id),
                **insn_options)

    from loopy.kernel.instruction import NoOpInstruction, BarrierInstruction
    special_insn_kind = groups["kind"]

    if special_insn_kind == "gbarrier":
        cls = BarrierInstruction
        kwargs["kind"] = "global"
    elif special_insn_kind == "lbarrier":
        cls = BarrierInstruction
        kwargs["kind"] = "local"
    elif special_insn_kind == "nop":
        cls = NoOpInstruction
    else:
        raise LoopyError(
            "invalid kind of special instruction: '%s'" % special_insn_kind)

    return cls(**kwargs), inames_to_dup

# }}}


# {{{ parse_instructions

_PAREN_PAIRS = {
        "(": (+1, "("),
        ")": (-1, "("),
        "[": (+1, "["),
        "]": (-1, "["),
        "{": (+1, "{"),
        "}": (-1, "}"),
        }


def _count_open_paren_symbols(s):
    result = 0
    for c in s:
        val = _PAREN_PAIRS.get(c)
        if val is not None:
            increment, cls = val
            result += increment

    return result


def parse_instructions(instructions, defines):
    if isinstance(instructions, str):
        instructions = [instructions]

    substitutions = {}

    new_instructions = []

    # {{{ pass 1: interning, comments, whitespace

    for insn in instructions:
        if isinstance(insn, SubstitutionRule):
            substitutions[insn.name] = insn
            continue

        elif isinstance(insn, InstructionBase):
            def intern_if_str(s):
                if isinstance(s, str):
                    return intern(s)
                else:
                    return s

            new_instructions.append(
                    insn.copy(
                        id=intern(insn.id) if isinstance(insn.id, str) else insn.id,
                        depends_on=frozenset(intern_if_str(dep)
                            for dep in insn.depends_on),
                        groups=frozenset(intern(grp) for grp in insn.groups),
                        conflicts_with_groups=frozenset(
                            intern(grp) for grp in insn.conflicts_with_groups),
                        within_inames=frozenset(
                            intern(iname) for iname in insn.within_inames),
                        ))
            continue

        elif not isinstance(insn, str):
            raise TypeError("Instructions must be either an Instruction "
                    "instance or a parseable string. got '%s' instead."
                    % type(insn))

        for insn in insn.split("\n"):
            comment_start = insn.find("#")
            if comment_start >= 0:
                insn = insn[:comment_start]

            insn = insn.strip()
            if not insn:
                continue

            new_instructions.append(insn)

    # }}}

    instructions = new_instructions
    new_instructions = []

    # {{{ pass 2: join-by-paren

    insn_buffer = None

    for i, insn in enumerate(instructions):
        if isinstance(insn, InstructionBase):
            if insn_buffer is not None:
                raise LoopyError("cannot join instruction lines "
                        "by paren-like delimiters "
                        "across InstructionBase instance at instructions index %d"
                        % i)

            new_instructions.append(insn)
        else:
            if insn_buffer is not None:
                insn_buffer = insn_buffer + " " + insn
                if _count_open_paren_symbols(insn_buffer) == 0:
                    new_instructions.append(insn_buffer)
                    insn_buffer = None

            else:
                if _count_open_paren_symbols(insn) == 0:
                    new_instructions.append(insn)
                else:
                    insn_buffer = insn

    if insn_buffer is not None:
        raise LoopyError("unclosed paren-like delimiter at end of 'instructions' "
                "while attempting to join lines by paren-like delimiters")

    # }}}

    instructions = new_instructions
    new_instructions = []

    # {{{ pass 3: defines

    for insn in instructions:
        if isinstance(insn, InstructionBase):
            new_instructions.append(insn)
        else:
            for sub_insn in expand_defines(insn, defines, single_valued=False):
                new_instructions.append(sub_insn)

    # }}}

    instructions = new_instructions
    new_instructions = []

    inames_to_dup = []  # one for each parsed_instruction

    # {{{ pass 4: parsing

    insn_options_stack = [get_default_insn_options_dict()]
    if_predicates_stack = [
            {'predicates': frozenset(),
                'insn_predicates': frozenset()}]

    for insn in instructions:
        if isinstance(insn, InstructionBase):
            local_w_inames = insn_options_stack[-1]["within_inames"]

            if insn.within_inames_is_final:
                if not (
                        local_w_inames <= insn.within_inames):
                    raise LoopyError("non-parsed instruction '%s' without "
                            "inames '%s' (but with final iname dependencies) "
                            "found inside 'for'/'with' block for inames "
                            "'%s'"
                            % (insn.id,
                                ", ".join(local_w_inames - insn.within_inames),
                                insn_options_stack[-1].within_inames))

            else:
                # not final, add inames from current scope
                kwargs = {}
                if insn.id is None:
                    kwargs["id"] = insn_options_stack[-1]["insn_id"]

                insn = insn.copy(
                        within_inames=insn.within_inames | local_w_inames,
                        within_inames_is_final=(
                            # If it's inside a for/with block, then it's
                            # final now.
                            bool(local_w_inames)),
                        depends_on=(
                            (insn.depends_on
                                | insn_options_stack[-1]["depends_on"])
                            if insn_options_stack[-1]["depends_on"] is not None
                            else insn.depends_on),
                        tags=(
                            insn.tags
                            | insn_options_stack[-1]["tags"]),
                        predicates=(
                            insn.predicates
                            | insn_options_stack[-1]["predicates"]),
                        groups=(
                            insn.groups
                            | insn_options_stack[-1]["groups"]),
                        conflicts_with_groups=(
                            insn.groups
                            | insn_options_stack[-1]["conflicts_with_groups"]),
                        **kwargs)

            new_instructions.append(insn)
            inames_to_dup.append([])

            del local_w_inames

            continue

        with_options_match = WITH_OPTIONS_RE.match(insn)
        if with_options_match is not None:
            insn_options_stack.append(
                    parse_insn_options(
                        insn_options_stack[-1],
                        with_options_match.group("options")))
            continue

        for_match = FOR_RE.match(insn)
        if for_match is not None:
            options = insn_options_stack[-1].copy()
            added_inames = frozenset(
                    iname.strip()
                    for iname in for_match.group("inames").split(",")
                    if iname.strip())
            if not added_inames:
                raise LoopyError("'for' without inames encountered")

            options["within_inames"] = (
                    options.get("within_inames", frozenset())
                    | added_inames)
            options["within_inames_is_final"] = True

            insn_options_stack.append(options)
            del options
            continue

        if_match = IF_RE.match(insn)
        if if_match is not None:
            options = insn_options_stack[-1].copy()
            predicate = if_match.group("predicate")
            if not predicate:
                raise LoopyError("'if' without predicate encountered")

            from loopy.symbolic import parse
            predicate = parse(predicate)

            options["predicates"] = (
                    options.get("predicates", frozenset())
                    | frozenset([predicate]))

            insn_options_stack.append(options)

            #add to the if_stack
            if_options = options.copy()
            if_options['insn_predicates'] = options["predicates"]
            if_predicates_stack.append(if_options)
            del options
            del predicate
            continue

        elif_match = ELIF_RE.match(insn)
        else_match = ELSE_RE.match(insn)
        if elif_match is not None or else_match is not None:
            prev_predicates = insn_options_stack[-1].get(
                    "predicates", frozenset())
            last_if_predicates = if_predicates_stack[-1].get(
                    "predicates", frozenset())
            insn_options_stack.pop()
            if_predicates_stack.pop()

            outer_predicates = insn_options_stack[-1].get(
                    "predicates", frozenset())
            last_if_predicates = last_if_predicates - outer_predicates

            if elif_match is not None:
                predicate = elif_match.group("predicate")
                if not predicate:
                    raise LoopyError("'elif' without predicate encountered")
                from loopy.symbolic import parse
                predicate = parse(predicate)

                additional_preds = frozenset([predicate])
                del predicate

            else:
                assert else_match is not None
                if not last_if_predicates:
                    raise LoopyError("'else' without 'if'/'elif' encountered")
                additional_preds = frozenset()

            options = insn_options_stack[-1].copy()
            if_options = insn_options_stack[-1].copy()

            from pymbolic.primitives import LogicalNot
            options["predicates"] = (
                    options.get("predicates", frozenset())
                    | outer_predicates
                    | prev_predicates - last_if_predicates
                    | frozenset(
                        LogicalNot(pred) for pred in last_if_predicates)
                    | additional_preds
                    )
            if_options["predicates"] = additional_preds
            #hold on to this for comparison / stack popping later
            if_options["insn_predicates"] = options["predicates"]

            insn_options_stack.append(options)
            if_predicates_stack.append(if_options)

            del options
            del additional_preds
            del last_if_predicates

            continue

        if insn == "end":
            obj = insn_options_stack.pop()
            #if this object is the end of an if statement
            if obj['predicates'] == if_predicates_stack[-1]["insn_predicates"] and\
                    if_predicates_stack[-1]["insn_predicates"]:
                if_predicates_stack.pop()
            continue

        insn_match = SPECIAL_INSN_RE.match(insn)
        if insn_match is not None:
            insn, insn_inames_to_dup = parse_special_insn(
                    insn_match.groupdict(), insn_options_stack[-1])
            new_instructions.append(insn)
            inames_to_dup.append(insn_inames_to_dup)
            continue

        subst_match = SUBST_RE.match(insn)
        if subst_match is not None:
            subst = parse_subst_rule(subst_match.groupdict())
            substitutions[subst.name] = subst
            continue

        insn_match = INSN_RE.match(insn)
        if insn_match is not None:
            insn, insn_inames_to_dup = parse_insn(
                    insn_match.groupdict(), insn_options_stack[-1])
            new_instructions.append(insn)
            inames_to_dup.append(insn_inames_to_dup)
            continue

        insn_match = EMPTY_LHS_INSN_RE.match(insn)
        if insn_match is not None:
            insn, insn_inames_to_dup = parse_insn(
                    insn_match.groupdict(), insn_options_stack[-1])
            new_instructions.append(insn)
            inames_to_dup.append(insn_inames_to_dup)
            continue

        raise LoopyError("instruction parse error: %s" % insn)

    if len(insn_options_stack) != 1:
        raise LoopyError("unbalanced number of 'for'/'with' and 'end' "
                "declarations")

    # }}}

    return new_instructions, inames_to_dup, substitutions

# }}}


# {{{ domain parsing

EMPTY_SET_DIMS_RE = re.compile(r"^\s*\{\s*\:")
SET_DIMS_RE = re.compile(r"^\s*\{\s*\[([a-zA-Z0-9_, ]+)\]\s*\:")


def _find_inames_in_set(dom_str):
    empty_match = EMPTY_SET_DIMS_RE.match(dom_str)
    if empty_match is not None:
        return set()

    match = SET_DIMS_RE.match(dom_str)
    if match is None:
        raise RuntimeError("invalid syntax for domain '%s'" % dom_str)

    result = set(iname.strip() for iname in match.group(1).split(",")
            if iname.strip())

    return result


EX_QUANT_RE = re.compile(r"\bexists\s+([a-zA-Z0-9])\s*\:")


def _find_existentially_quantified_inames(dom_str):
    return set(ex_quant.group(1) for ex_quant in EX_QUANT_RE.finditer(dom_str))


def parse_domains(domains, defines):
    if isinstance(domains, str):
        domains = [domains]

    result = []
    used_inames = set()

    for dom in domains:
        if isinstance(dom, str):
            dom, = expand_defines(dom, defines)

            if not dom.lstrip().startswith("["):
                # i.e. if no parameters are already given
                parameters = (_gather_isl_identifiers(dom)
                        - _find_inames_in_set(dom)
                        - _find_existentially_quantified_inames(dom))
                dom = "[%s] -> %s" % (",".join(sorted(parameters)), dom)

            try:
                dom = isl.BasicSet.read_from_str(isl.DEFAULT_CONTEXT, dom)
            except:
                print("failed to parse domain '%s'" % dom)
                raise
        else:
            assert isinstance(dom, (isl.Set, isl.BasicSet))
            # assert dom.get_ctx() == ctx

        if isinstance(dom, isl.Set):
            from loopy.isl_helpers import convexify
            dom = convexify(dom)

        for i_iname in range(dom.dim(dim_type.set)):
            iname = dom.get_dim_name(dim_type.set, i_iname)

            if iname is None:
                raise RuntimeError("domain '%s' provided no iname at index "
                        "%d (redefined iname?)" % (dom, i_iname))

            if iname in used_inames:
                raise RuntimeError("domain '%s' redefines iname '%s' "
                        "that is part of a previous domain" % (dom, iname))

            used_inames.add(iname)

        result.append(dom)

    return result

# }}}


# {{{ guess kernel args (if requested)

class IndexRankFinder(CSECachingMapperMixin, WalkMapper):
    def __init__(self, arg_name):
        self.arg_name = arg_name
        self.index_ranks = []

    def map_subscript(self, expr):
        WalkMapper.map_subscript(self, expr)

        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        if expr.aggregate.name == self.arg_name:
            if not isinstance(expr.index, tuple):
                self.index_ranks.append(1)
            else:
                self.index_ranks.append(len(expr.index))

    def map_common_subexpression_uncached(self, expr):
        if not self.visit(expr):
            return

        self.rec(expr.child)
        self.post_visit(expr)


class ArgumentGuesser:
    def __init__(self, domains, instructions, temporary_variables,
            subst_rules, default_offset):
        self.domains = domains
        self.instructions = instructions
        self.temporary_variables = temporary_variables
        self.subst_rules = subst_rules
        self.default_offset = default_offset

        from loopy.symbolic import SubstitutionRuleExpander
        self.submap = SubstitutionRuleExpander(subst_rules)

        self.all_inames = set()
        for dom in domains:
            self.all_inames.update(dom.get_var_names(dim_type.set))

        all_params = set()
        for dom in domains:
            all_params.update(dom.get_var_names(dim_type.param))
        self.all_params = all_params - self.all_inames

        self.all_names = set()
        self.all_written_names = set()
        from loopy.symbolic import get_dependencies
        for insn in instructions:
            if isinstance(insn, MultiAssignmentBase):
                for assignee_var_name in insn.assignee_var_names():
                    self.all_written_names.add(assignee_var_name)

                self.all_names.update(get_dependencies(
                    self.submap(insn.assignees)))
                self.all_names.update(get_dependencies(
                    self.submap(insn.expression)))

    def find_index_rank(self, name):
        irf = IndexRankFinder(name)

        def run_irf(expr):
            irf(self.submap(expr))
            return expr

        for insn in self.instructions:
            insn.with_transformed_expressions(run_irf)

        if not irf.index_ranks:
            return 0
        else:
            from pytools import single_valued
            return single_valued(irf.index_ranks)

    def make_new_arg(self, arg_name):
        arg_name = arg_name.strip()

        from loopy.kernel.data import ValueArg, GlobalArg
        import loopy as lp

        if arg_name in self.all_params:
            return ValueArg(arg_name)

        if arg_name in self.all_written_names:
            # It's not a temp var, and thereby not a domain parameter--the only
            # other writable type of variable is an argument.

            return GlobalArg(arg_name,
                    shape=lp.auto, offset=self.default_offset)

        irank = self.find_index_rank(arg_name)
        if irank == 0:
            # read-only, no indices
            return ValueArg(arg_name)
        else:
            return GlobalArg(arg_name, shape=lp.auto, offset=self.default_offset)

    def convert_names_to_full_args(self, kernel_args):
        new_kernel_args = []

        for arg in kernel_args:
            if isinstance(arg, str) and arg != "...":
                new_kernel_args.append(self.make_new_arg(arg))
            else:
                new_kernel_args.append(arg)

        return new_kernel_args

    def guess_kernel_args_if_requested(self, kernel_args):
        # Ellipsis is syntactically allowed in Py3.
        if "..." not in kernel_args and Ellipsis not in kernel_args:
            return kernel_args

        kernel_args = [arg for arg in kernel_args
                if arg is not Ellipsis and arg != "..."]

        # {{{ find names that are *not* arguments

        temp_var_names = set(six.iterkeys(self.temporary_variables))

        for insn in self.instructions:
            if isinstance(insn, MultiAssignmentBase):
                for assignee_var_name, temp_var_type in zip(
                        insn.assignee_var_names(),
                        insn.temp_var_types):
                    if temp_var_type is not None:
                        temp_var_names.add(assignee_var_name)

        # }}}

        # {{{ find existing and new arg names

        existing_arg_names = set()
        for arg in kernel_args:
            existing_arg_names.add(arg.name)

        not_new_arg_names = existing_arg_names | temp_var_names | self.all_inames

        from loopy.kernel.data import ArrayBase
        from loopy.symbolic import get_dependencies
        for arg in kernel_args:
            if isinstance(arg, ArrayBase):
                if isinstance(arg.shape, tuple):
                    self.all_names.update(
                            get_dependencies(arg.shape))

        new_arg_names = (self.all_names | self.all_params) - not_new_arg_names

        # }}}

        for arg_name in sorted(new_arg_names):
            kernel_args.append(self.make_new_arg(arg_name))

        return kernel_args

# }}}


# {{{ sanity checking

def check_for_duplicate_names(knl):
    name_to_source = {}

    def add_name(name, source):
        if name in name_to_source:
            raise RuntimeError("invalid %s name '%s'--name already used as "
                    "%s" % (source, name, name_to_source[name]))

        name_to_source[name] = source

    for name in knl.all_inames():
        add_name(name, "iname")
    for arg in knl.args:
        add_name(arg.name, "argument")
    for name in knl.temporary_variables:
        add_name(name, "temporary")
    for name in knl.substitutions:
        add_name(name, "substitution")


def check_for_nonexistent_iname_deps(knl):
    for insn in knl.instructions:
        if not set(insn.within_inames) <= knl.all_inames():
            raise ValueError("In instruction '%s': "
                    "cannot force dependency on inames '%s'--"
                    "they don't exist" % (
                        insn.id,
                        ",".join(
                            set(insn.within_inames)-knl.all_inames())))


def check_for_multiple_writes_to_loop_bounds(knl):
    from islpy import dim_type

    domain_parameters = set()
    for dom in knl.domains:
        domain_parameters.update(dom.get_space().get_var_dict(dim_type.param))

    temp_var_domain_parameters = domain_parameters & set(
            knl.temporary_variables)

    wmap = knl.writer_map()
    for tvpar in temp_var_domain_parameters:
        par_writers = wmap[tvpar]
        if len(par_writers) != 1:
            raise RuntimeError("there must be exactly one write "
                    "to data-dependent domain parameter '%s' (found %d)"
                    % (tvpar, len(par_writers)))


def check_written_variable_names(knl):
    admissible_vars = (
            set(arg.name for arg in knl.args)
            | set(six.iterkeys(knl.temporary_variables)))

    for insn in knl.instructions:
        for var_name in insn.assignee_var_names():
            if var_name not in admissible_vars:
                raise RuntimeError("variable '%s' not declared or not "
                        "allowed for writing" % var_name)

# }}}


# {{{ expand common subexpressions into assignments

class CSEToAssignmentMapper(IdentityMapper):
    def __init__(self, add_assignment):
        self.add_assignment = add_assignment
        self.expr_to_var = {}

    def map_reduction(self, expr, additional_inames):
        additional_inames = additional_inames | frozenset(expr.inames)

        return super(CSEToAssignmentMapper, self).map_reduction(
                expr, additional_inames)

    def map_common_subexpression(self, expr, additional_inames):
        try:
            return self.expr_to_var[expr.child]
        except KeyError:
            from loopy.symbolic import TypedCSE
            if isinstance(expr, TypedCSE):
                dtype = expr.dtype
            else:
                dtype = None

            child = self.rec(expr.child, additional_inames)
            from pymbolic.primitives import Variable
            if isinstance(child, Variable):
                return child

            var_name = self.add_assignment(
                    expr.prefix, child, dtype, additional_inames)
            var = Variable(var_name)
            self.expr_to_var[expr.child] = var
            return var


def expand_cses(instructions, inames_to_dup, cse_prefix="cse_expr"):
    def add_assignment(base_name, expr, dtype, additional_inames):
        if base_name is None:
            base_name = "var"

        new_var_name = var_name_gen(base_name)

        if dtype is None:
            import loopy as lp
            dtype = lp.auto
        else:
            dtype = np.dtype(dtype)

        from loopy.kernel.data import TemporaryVariable
        new_temp_vars.append(TemporaryVariable(
                name=new_var_name,
                dtype=dtype,
                scope=lp.auto,
                shape=()))

        from pymbolic.primitives import Variable
        new_insn = Assignment(
                id=None,
                assignee=Variable(new_var_name),
                expression=expr,
                predicates=insn.predicates,
                within_inames=insn.within_inames | additional_inames,
                within_inames_is_final=insn.within_inames_is_final,
                )
        newly_created_insn_ids.add(new_insn.id)
        new_insns.append(new_insn)
        if insn_inames_to_dup:
            raise LoopyError("in-line iname duplication not allowed in "
                    "an instruction containing a tagged common "
                    "subexpression (found in instruction '%s')"
                    % insn)

        new_inames_to_dup.append(insn_inames_to_dup)

        return new_var_name

    cseam = CSEToAssignmentMapper(add_assignment=add_assignment)

    new_insns = []
    new_inames_to_dup = []

    from pytools import UniqueNameGenerator
    var_name_gen = UniqueNameGenerator(forced_prefix=cse_prefix)

    newly_created_insn_ids = set()
    new_temp_vars = []

    for insn, insn_inames_to_dup in zip(instructions, inames_to_dup):
        if isinstance(insn, MultiAssignmentBase):
            new_insns.append(insn.copy(
                expression=cseam(insn.expression, frozenset())))
            new_inames_to_dup.append(insn_inames_to_dup)
        else:
            new_insns.append(insn)
            new_inames_to_dup.append(insn_inames_to_dup)

    return new_insns, new_inames_to_dup, new_temp_vars

# }}}


# {{{ add_sequential_dependencies

def add_sequential_dependencies(knl):
    new_insns = []
    prev_insn = None
    for insn in knl.instructions:
        depon = insn.depends_on
        if depon is None:
            depon = frozenset()

        if prev_insn is not None:
            depon = depon | frozenset((prev_insn.id,))

        insn = insn.copy(
                depends_on=depon,
                depends_on_is_final=True)

        new_insns.append(insn)

        prev_insn = insn

    return knl.copy(instructions=new_insns)

# }}}


# {{{ temporary variable creation

def create_temporaries(knl, default_order):
    new_insns = []
    new_temp_vars = knl.temporary_variables.copy()

    import loopy as lp

    for insn in knl.instructions:
        if isinstance(insn, MultiAssignmentBase):
            for assignee_name, temp_var_type in zip(
                    insn.assignee_var_names(),
                    insn.temp_var_types):

                if temp_var_type is None:
                    continue

                if assignee_name in new_temp_vars:
                    raise RuntimeError("cannot create temporary variable '%s'--"
                            "already exists" % assignee_name)
                if assignee_name in knl.arg_dict:
                    raise RuntimeError("cannot create temporary variable '%s'--"
                            "already exists as argument" % assignee_name)

                logger.debug("%s: creating temporary %s"
                        % (knl.name, assignee_name))

                new_temp_vars[assignee_name] = lp.TemporaryVariable(
                        name=assignee_name,
                        dtype=temp_var_type,
                        scope=lp.auto,
                        base_indices=lp.auto,
                        shape=lp.auto,
                        order=default_order,
                        target=knl.target)

                if isinstance(insn, Assignment):
                    insn = insn.copy(temp_var_type=None)
                else:
                    insn = insn.copy(temp_var_types=None)

        new_insns.append(insn)

    return knl.copy(
            instructions=new_insns,
            temporary_variables=new_temp_vars)

# }}}


# {{{ determine shapes of temporaries

def find_shapes_of_vars(knl, var_names, feed_expression):
    from loopy.symbolic import BatchedAccessRangeMapper, SubstitutionRuleExpander
    submap = SubstitutionRuleExpander(knl.substitutions)

    armap = BatchedAccessRangeMapper(knl, var_names)

    def run_through_armap(expr, inames):
        armap(submap(expr), inames)
        return expr

    feed_expression(run_through_armap)

    var_to_base_indices = {}
    var_to_shape = {}
    var_to_error = {}

    from loopy.diagnostic import StaticValueFindingError

    for var_name in var_names:
        access_range = armap.access_ranges[var_name]
        bad_subscripts = armap.bad_subscripts[var_name]

        if access_range is not None:
            try:
                base_indices, shape = list(zip(*[
                        knl.cache_manager.base_index_and_length(
                            access_range, i)
                        for i in range(access_range.dim(dim_type.set))]))
            except StaticValueFindingError as e:
                var_to_error[var_name] = str(e)
                continue

        else:
            if bad_subscripts:
                raise RuntimeError("cannot determine access range for '%s': "
                        "undetermined index in subscript(s) '%s'"
                        % (var_name, ", ".join(
                                str(i) for i in bad_subscripts)))

            # no subscripts found, let's call it a scalar
            base_indices = ()
            shape = ()

        var_to_base_indices[var_name] = base_indices
        var_to_shape[var_name] = shape

    return var_to_base_indices, var_to_shape, var_to_error


def determine_shapes_of_temporaries(knl):
    new_temp_vars = knl.temporary_variables.copy()

    import loopy as lp

    vars_needing_shape_inference = set()

    for tv in six.itervalues(knl.temporary_variables):
        if tv.shape is lp.auto or tv.base_indices is lp.auto:
            vars_needing_shape_inference.add(tv.name)

    def feed_all_expressions(receiver):
        for insn in knl.instructions:
            insn.with_transformed_expressions(
                lambda expr: receiver(expr, knl.insn_inames(insn)))

    var_to_base_indices, var_to_shape, var_to_error = (
        find_shapes_of_vars(
                knl, vars_needing_shape_inference, feed_all_expressions))

    # {{{ fall back to legacy method

    if len(var_to_error) > 0:
        vars_needing_shape_inference = set(var_to_error.keys())

        from six import iteritems
        for varname, err in iteritems(var_to_error):
            warn_with_kernel(knl, "temp_shape_fallback",
                             "Had to fall back to legacy method of determining "
                             "shape of temporary '%s' because: %s"
                             % (varname, err))

        def feed_assignee_of_instruction(receiver):
            for insn in knl.instructions:
                for assignee in insn.assignees:
                    receiver(assignee, knl.insn_inames(insn))

        var_to_base_indices_fallback, var_to_shape_fallback, var_to_error = (
            find_shapes_of_vars(
                    knl, vars_needing_shape_inference, feed_assignee_of_instruction))

        if len(var_to_error) > 0:
            # No way around errors: propagate an exception upward.
            formatted_errors = (
                "\n\n".join("'%s': %s" % (varname, var_to_error[varname])
                for varname in sorted(var_to_error.keys())))

            raise LoopyError("got the following exception(s) trying to find the "
                    "shape of temporary variables: %s" % formatted_errors)

        var_to_base_indices.update(var_to_base_indices_fallback)
        var_to_shape.update(var_to_shape_fallback)

    # }}}

    new_temp_vars = {}

    for tv in six.itervalues(knl.temporary_variables):
        if tv.base_indices is lp.auto:
            tv = tv.copy(base_indices=var_to_base_indices[tv.name])
        if tv.shape is lp.auto:
            tv = tv.copy(shape=var_to_shape[tv.name])
        new_temp_vars[tv.name] = tv

    return knl.copy(temporary_variables=new_temp_vars)

# }}}


# {{{ expand defines in shapes

def expand_defines_in_shapes(kernel, defines):
    from loopy.kernel.array import ArrayBase
    from loopy.kernel.creation import expand_defines_in_expr

    def expr_map(expr):
        return expand_defines_in_expr(expr, defines)

    processed_args = []
    for arg in kernel.args:
        if isinstance(arg, ArrayBase):
            arg = arg.map_exprs(expr_map)

        processed_args.append(arg)

    processed_temp_vars = {}
    for tv in six.itervalues(kernel.temporary_variables):
        processed_temp_vars[tv.name] = tv.map_exprs(expr_map)

    return kernel.copy(
            args=processed_args,
            temporary_variables=processed_temp_vars,
            )

# }}}


# {{{ guess argument shapes

def guess_arg_shape_if_requested(kernel, default_order):
    new_args = []

    import loopy as lp
    from loopy.kernel.array import ArrayBase
    from loopy.kernel.tools import guess_var_shape

    for arg in kernel.args:
        if isinstance(arg, ArrayBase) and arg.shape is lp.auto:
            shape = guess_var_shape(kernel, arg.name)

            if arg.shape is lp.auto:
                arg = arg.copy(shape=shape)

            try:
                arg.strides
            except AttributeError:
                pass
            else:
                if arg.strides is lp.auto:
                    from loopy.kernel.data import make_strides
                    arg = arg.copy(strides=make_strides(shape, default_order))

        new_args.append(arg)

    return kernel.copy(args=new_args)

# }}}


# {{{ apply default_order to args

def apply_default_order_to_args(kernel, default_order):
    from loopy.kernel.array import ArrayBase

    processed_args = []
    for arg in kernel.args:
        if isinstance(arg, ArrayBase) and arg.order is None:
            arg = arg.copy(order=default_order)
        processed_args.append(arg)

    return kernel.copy(args=processed_args)

# }}}


# {{{ resolve instruction dependencies

def _resolve_dependencies(knl, insn, deps):
    from loopy import find_instructions
    from loopy.match import MatchExpressionBase

    new_deps = []

    for dep in deps:
        found_any = False

        if isinstance(dep, MatchExpressionBase):
            for new_dep in find_instructions(knl, dep):
                if new_dep.id != insn.id:
                    new_deps.append(new_dep.id)
                    found_any = True
        else:
            from fnmatch import fnmatchcase
            for other_insn in knl.instructions:
                if fnmatchcase(other_insn.id, dep):
                    new_deps.append(other_insn.id)
                    found_any = True

        if not found_any and knl.options.check_dep_resolution:
            raise LoopyError("instruction '%s' declared a depency on '%s', "
                    "which did not resolve to any instruction present in the "
                    "kernel '%s'. Set the kernel option 'check_dep_resolution'"
                    "to False to disable this check." % (insn.id, dep, knl.name))

    for dep_id in new_deps:
        if dep_id not in knl.id_to_insn:
            raise LoopyError("instruction '%s' depends on instruction id '%s', "
                    "which was not found" % (insn.id, dep_id))

    return frozenset(new_deps)


def resolve_dependencies(knl):
    new_insns = []

    for insn in knl.instructions:
        new_insns.append(insn.copy(
                    depends_on=_resolve_dependencies(knl, insn, insn.depends_on),
                    no_sync_with=frozenset(
                        (resolved_insn_id, nosync_scope)
                        for nosync_dep, nosync_scope in insn.no_sync_with
                        for resolved_insn_id in
                        _resolve_dependencies(knl, insn, (nosync_dep,))),
                    ))

    return knl.copy(instructions=new_insns)

# }}}


# {{{ add used inames deps

def add_used_inames(knl):
    new_insns = []

    for insn in knl.instructions:
        deps = insn.read_dependency_names() | insn.write_dependency_names()
        iname_deps = deps & knl.all_inames()

        new_within_inames = insn.within_inames | iname_deps

        if new_within_inames != insn.within_inames:
            insn = insn.copy(within_inames=new_within_inames)

        new_insns.append(insn)

    return knl.copy(instructions=new_insns)

# }}}


# {{{ add inferred iname deps

def add_inferred_inames(knl):
    from loopy.kernel.tools import find_all_insn_inames
    insn_inames = find_all_insn_inames(knl)

    return knl.copy(instructions=[
            insn.copy(within_inames=insn_inames[insn.id])
            for insn in knl.instructions])

# }}}


# {{{ apply single-writer heuristic

def apply_single_writer_depencency_heuristic(kernel, warn_if_used=True):
    logger.debug("%s: default deps" % kernel.name)

    from loopy.transform.subst import expand_subst
    expanded_kernel = expand_subst(kernel)

    writer_map = kernel.writer_map()

    arg_names = set(arg.name for arg in kernel.args)

    var_names = arg_names | set(six.iterkeys(kernel.temporary_variables))

    dep_map = dict(
            (insn.id, insn.read_dependency_names() & var_names)
            for insn in expanded_kernel.instructions)

    new_insns = []
    for insn in kernel.instructions:
        if not insn.depends_on_is_final:
            auto_deps = set()

            # {{{ add automatic dependencies

            all_my_var_writers = set()
            for var in dep_map[insn.id]:
                var_writers = writer_map.get(var, set())
                all_my_var_writers |= var_writers

                if not var_writers and var not in arg_names:
                    tv = kernel.temporary_variables[var]
                    if tv.initializer is None:
                        warn_with_kernel(kernel, "read_no_write(%s)" % var,
                                "temporary variable '%s' is read, but never written."
                                % var)

                if len(var_writers) == 1:
                    auto_deps.update(
                            var_writers
                            - set([insn.id]))

            # }}}

            depends_on = insn.depends_on
            if depends_on is None:
                depends_on = frozenset()

            new_deps = frozenset(auto_deps) | depends_on

            if warn_if_used and new_deps != depends_on:
                warn_with_kernel(kernel, "single_writer_after_creation",
                        "The single-writer dependency heuristic added dependencies "
                        "on instruction ID(s) '%s' to instruction ID '%s' after "
                        "kernel creation is complete. This is deprecated and "
                        "may stop working in the future. "
                        "To fix this, ensure that instruction dependencies "
                        "are added/resolved as soon as possible, ideally at kernel "
                        "creation time."
                        % (", ".join(new_deps - depends_on), insn.id))

            insn = insn.copy(depends_on=new_deps)

        new_insns.append(insn)

    return kernel.copy(instructions=new_insns)

# }}}


# {{{ kernel creation top-level

def make_kernel(domains, instructions, kernel_data=["..."], **kwargs):
    """User-facing kernel creation entrypoint.

    :arg domains:

        A list of :class:`islpy.BasicSet` (i.e. convex set) instances
        representing the :ref:`domain-tree`. May also be a list of strings
        which will be parsed into such instances according to :ref:`isl-syntax`.

    :arg instructions:

        A list of :class:`Assignment` (or other :class:`InstructionBase`
        subclasses), possibly intermixed with instances of
        :class:`SubstitutionRule`. This same list may also contain strings
        which will be parsed into such objects using the
        :ref:`assignment-syntax` and the :ref:`subst-rule-syntax`.  May also be
        a single multi-line string which will be split into lines and then
        parsed.

    :arg kernel_data:

        A list of :class:`ValueArg`, :class:`GlobalArg`, ... (etc.) instances.
        The order of these arguments determines the order of the arguments
        to the generated kernel.

        May also contain :class:`TemporaryVariable` instances(which do not
        give rise to kernel-level arguments).

        The string ``"..."`` may be passed as one of the entries
        of the list, in which case loopy will infer names, shapes,
        and types of arguments from the kernel code. It is
        possible to just pass the list ``["..."]``, in which case
        all arguments are inferred.

        In Python 3, the string ``"..."`` may be spelled somewhat more sensibly
        as just ``...`` (the ellipsis), for the same meaning.

        As an additional option, each argument may be specified as just a name
        (a string). This is useful to specify argument ordering. All other
        characteristics of the named arguments are inferred.

    The following keyword arguments are recognized:

    :arg preambles: a list of (tag, code) tuples that identify preamble
        snippets.
        Each tag's snippet is only included once, at its first occurrence.
        The preambles will be inserted in order of their tags.
    :arg preamble_generators: a list of functions of signature
        (seen_dtypes, seen_functions) where seen_functions is a set of
        (name, c_name, arg_dtypes), generating extra entries for *preambles*.
    :arg default_order: "C" (default) or "F"
    :arg default_offset: 0 or :class:`loopy.auto`. The default value of
        *offset* in :attr:`GlobalArg` for guessed arguments.
        Defaults to 0.
    :arg function_manglers: list of functions of signature
        ``(target, name, arg_dtypes)``
        returning a :class:`loopy.CallMangleInfo`.
    :arg symbol_manglers: list of functions of signature (name) returning
        a tuple (result_dtype, c_name), where c_name is the C-level symbol to
        be evaluated.
    :arg assumptions: the initial implemented_domain, captures assumptions
        on loop domain parameters. (an isl.Set or a string in
        :ref:`isl-syntax`.  If given as a string, only the CONDITIONS part of
        the set notation should be given.)
    :arg local_sizes: A dictionary from integers to integers, mapping
        workgroup axes to their sizes, e.g. *{0: 16}* forces axis 0 to be
        length 16.
    :arg silenced_warnings: a list (or semicolon-separated string) or warnings
        to silence
    :arg options: an instance of :class:`loopy.Options` or an equivalent
        string representation
    :arg target: an instance of :class:`loopy.TargetBase`, or *None*,
        to use the default target.
    :arg seq_dependencies: If *True*, dependencies that sequentially
        connect the given *instructions* will be added. Defaults to
        *False*.
    :arg fixed_parameters: A dictionary of *name*/*value* pairs, where *name*
        will be fixed to *value*. *name* may refer to :ref:`domain-parameters`
        or :ref:`arguments`. See also :func:`loopy.fix_parameters`.

    .. versionchanged:: 2017.2

        *fixed_parameters* added.

    .. versionchanged:: 2016.3

        *seq_dependencies* added.
    """

    logger.info(
            "%s: kernel creation start" % kwargs.get("name", "(unnamed)"))

    defines = kwargs.pop("defines", {})
    default_order = kwargs.pop("default_order", "C")
    default_offset = kwargs.pop("default_offset", 0)
    silenced_warnings = kwargs.pop("silenced_warnings", [])
    options = kwargs.pop("options", None)
    flags = kwargs.pop("flags", None)
    target = kwargs.pop("target", None)
    seq_dependencies = kwargs.pop("seq_dependencies", False)
    fixed_parameters = kwargs.pop("fixed_parameters", {})

    if defines:
        from warnings import warn
        warn("'defines' argument to make_kernel is deprecated. "
                "Use lp.fix_parameters instead",
                DeprecationWarning, stacklevel=2)

    if target is None:
        from loopy import _DEFAULT_TARGET
        target = _DEFAULT_TARGET

    if flags is not None:
        if options is not None:
            raise TypeError("may not pass both 'options' and 'flags'")

        from warnings import warn
        warn("'flags' is deprecated. Use 'options' instead",
                DeprecationWarning, stacklevel=2)
        options = flags

    from loopy.options import make_options
    options = make_options(options)

    if isinstance(silenced_warnings, str):
        silenced_warnings = silenced_warnings.split(";")

    # {{{ separate temporary variables and arguments, take care of names with commas

    from loopy.kernel.data import TemporaryVariable, ArrayBase

    if isinstance(kernel_data, str):
        kernel_data = kernel_data.split(",")

    kernel_args = []
    temporary_variables = kwargs.pop("temporary_variables", {}).copy()
    for dat in kernel_data:
        if dat is Ellipsis or isinstance(dat, str):
            kernel_args.append(dat)
            continue

        if isinstance(dat, ArrayBase) and isinstance(dat.shape, tuple):
            new_shape = []
            for shape_axis in dat.shape:
                if shape_axis is not None:
                    new_shape.append(expand_defines_in_expr(shape_axis, defines))
                else:
                    new_shape.append(shape_axis)
            dat = dat.copy(shape=tuple(new_shape))

        for arg_name in dat.name.split(","):
            arg_name = arg_name.strip()
            if not arg_name:
                continue

            my_dat = dat.copy(name=arg_name)
            if isinstance(dat, TemporaryVariable):
                temporary_variables[my_dat.name] = dat
            else:
                kernel_args.append(my_dat)

    del kernel_data

    # }}}

    instructions, inames_to_dup, substitutions = \
            parse_instructions(instructions, defines)

    # {{{ find/create isl_context

    for domain in domains:
        if isinstance(domain, isl.BasicSet):
            assert domain.get_ctx() == isl.DEFAULT_CONTEXT

    # }}}

    instructions, inames_to_dup, cse_temp_vars = expand_cses(
            instructions, inames_to_dup)
    for tv in cse_temp_vars:
        temporary_variables[tv.name] = tv
    del cse_temp_vars

    domains = parse_domains(domains, defines)

    arg_guesser = ArgumentGuesser(domains, instructions,
            temporary_variables, substitutions,
            default_offset)

    kernel_args = arg_guesser.convert_names_to_full_args(kernel_args)
    kernel_args = arg_guesser.guess_kernel_args_if_requested(kernel_args)

    kwargs["substitutions"] = substitutions

    from loopy.kernel import LoopKernel
    knl = LoopKernel(domains, instructions, kernel_args,
            temporary_variables=temporary_variables,
            silenced_warnings=silenced_warnings,
            options=options,
            target=target,
            **kwargs)

    if seq_dependencies:
        knl = add_sequential_dependencies(knl)

    assert len(knl.instructions) == len(inames_to_dup)

    from loopy import duplicate_inames
    from loopy.match import Id
    for insn, insn_inames_to_dup in zip(knl.instructions, inames_to_dup):
        for old_iname, new_iname in insn_inames_to_dup:
            knl = duplicate_inames(knl, old_iname,
                    within=Id(insn.id), new_inames=new_iname)

    check_for_nonexistent_iname_deps(knl)

    knl = create_temporaries(knl, default_order)
    # -------------------------------------------------------------------------
    # Ordering dependency:
    # -------------------------------------------------------------------------
    # Must create temporaries before inferring inames (because those temporaries
    # mediate dependencies that are then used for iname propagation.)
    # Must create temporaries before fixing parameters.
    # -------------------------------------------------------------------------
    knl = add_used_inames(knl)
    # NOTE: add_inferred_inames will be phased out and throws warnings if it
    # does something.
    knl = add_inferred_inames(knl)
    from loopy.transform.parameter import fix_parameters
    knl = fix_parameters(knl, **fixed_parameters)
    # -------------------------------------------------------------------------
    # Ordering dependency:
    # -------------------------------------------------------------------------
    # Must infer inames before determining shapes.
    # -------------------------------------------------------------------------
    knl = determine_shapes_of_temporaries(knl)

    knl = expand_defines_in_shapes(knl, defines)
    knl = guess_arg_shape_if_requested(knl, default_order)
    knl = apply_default_order_to_args(knl, default_order)
    knl = resolve_dependencies(knl)
    knl = apply_single_writer_depencency_heuristic(knl, warn_if_used=False)

    # -------------------------------------------------------------------------
    # Ordering dependency:
    # -------------------------------------------------------------------------
    # Must create temporaries before checking for writes to temporary variables
    # that are domain parameters.
    # -------------------------------------------------------------------------

    check_for_multiple_writes_to_loop_bounds(knl)
    check_for_duplicate_names(knl)
    check_written_variable_names(knl)

    from loopy.preprocess import prepare_for_caching
    knl = prepare_for_caching(knl)

    logger.info(
            "%s: kernel creation done" % knl.name)

    return knl

# }}}

# vim: fdm=marker

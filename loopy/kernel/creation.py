"""UI for kernel creation."""


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
from pymbolic.primitives import Slice, Variable, Subscript, Call
from loopy.kernel.array import FixedStrideArrayDimTag
from loopy.tools import intern_frozenset_of_ids, Optional
from loopy.symbolic import (
        IdentityMapper, WalkMapper, SubArrayRef)
from loopy.kernel.data import (
        InstructionBase,
        MultiAssignmentBase, Assignment,
        SubstitutionRule, AddressSpace, ValueArg, auto)
from loopy.translation_unit import for_each_kernel
from loopy.diagnostic import LoopyError, warn_with_kernel
import islpy as isl
from islpy import dim_type
from pytools import ProcessLogger

from sys import intern

import re

import logging
logger = logging.getLogger(__name__)


# {{{ identifier wrangling

_IDENTIFIER_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")

# source: check_keywords() in isl_stream.c, ISL version 0.17
_ISL_KEYWORDS = frozenset("""
        exists and or implies not infty infinity NaN min max rat true false ceild
        floord mod ceil floor""".split())


def _gather_isl_identifiers(s):
    return set(_IDENTIFIER_RE.findall(s)) - _ISL_KEYWORDS


class UniqueName:
    """A tag for a string that identifies a partial identifier that is to
    be made unique by the UI.
    """

    def __init__(self, name):
        self.name = name

# }}}


# {{{ tag normalization

def _normalize_string_tag(tag):
    from pytools.tag import Tag

    from loopy.kernel.instruction import (
            UseStreamingStoreTag, LegacyStringInstructionTag)
    if tag == "!streaming_store":
        return UseStreamingStoreTag()
    else:
        from pytools import resolve_name
        try:
            tag_cls = resolve_name(tag)
        except ImportError:
            pass
        except AttributeError:
            pass
        else:
            if issubclass(tag_cls, Tag):
                return tag_cls()

        return LegacyStringInstructionTag(tag)


def _normalize_tags(tags):
    return frozenset(
                    _normalize_string_tag(t) if isinstance(t, str) else t
                    for t in tags)

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
    if not defines:
        return expr

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
                (scope, ", ".join("'%s'" % s for s in allowable_scopes)))
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
                    [s for s in opt_value.split(":") if s])

        elif opt_key == "if" and opt_value is not None:
            predicates = opt_value.split(":")
            new_predicates = set(result["predicates"])

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
            result["tags"] = _normalize_tags([
                    tag.strip() for tag in opt_value.split(":")
                    if tag.strip()])

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

        elif opt_key == "mem_kind":
            opt_value = opt_value.lower().strip()
            if opt_value not in ["local", "global"]:
                raise LoopyError("Unknown memory synchronization type %s specified"
                    " expected, 'local' or 'global'."
                    % opt_value)
            result["mem_kind"] = opt_value

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


def check_illegal_options(insn_options, insn_type):
    illegal_options = []
    if insn_type not in ["gbarrier", "lbarrier"]:
        illegal_options.append("mem_kind")

    bad_options = [x for x in illegal_options if x in insn_options]
    if bad_options:
        raise LoopyError("Cannot supply option(s) '%s' to instruction type '%s'" %
                         ", ".join(bad_options), insn_type)


def parse_insn(groups, insn_options):
    """
    :return: a tuple ``(insn, inames_to_dup)``, where insn is a
        :class:`Assignment`, a :class:`CallInstruction`,
        or a :class:`SubstitutionRule`
        and *inames_to_dup* is None or a list of tuples `(old, new)`.
    """

    from loopy.symbolic import parse

    if "lhs" in groups:
        try:
            lhs = parse(groups["lhs"])
        except Exception:
            print("While parsing left hand side '%s', "
                    "the following error occurred:" % groups["lhs"])
            raise
    else:
        lhs = ()

    try:
        rhs = parse(groups["rhs"])
    except Exception:
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
            assert isinstance(lhs_i.type, Optional)
            temp_var_types.append(lhs_i.type)
            lhs_i = lhs_i.child
        else:
            temp_var_types.append(Optional())

        inner_lhs_i = lhs_i
        if isinstance(inner_lhs_i, Lookup):
            inner_lhs_i = inner_lhs_i.aggregate

        from loopy.symbolic import LinearSubscript
        if isinstance(inner_lhs_i, Variable):
            assignee_names.append(inner_lhs_i.name)
        elif isinstance(inner_lhs_i, (Subscript, LinearSubscript)):
            assignee_names.append(inner_lhs_i.aggregate.name)
        elif isinstance(inner_lhs_i, SubArrayRef):
            assignee_names.append(inner_lhs_i.subscript.aggregate.name)
        else:
            raise LoopyError("left hand side of assignment '%s' must "
                    "be variable, subscript or a SubArrayRef" % (lhs_i,))

        new_lhs.append(lhs_i)

    lhs = tuple(new_lhs)
    temp_var_types = tuple(temp_var_types)
    del new_lhs

    insn_options = parse_insn_options(
            insn_options.copy(),
            groups["options"],
            assignee_names=assignee_names)

    # check for bad options
    check_illegal_options(insn_options, "assignment")

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
    except Exception:
        print("While parsing left hand side '%s', "
                "the following error occurred:" % groups["lhs"])
        raise

    try:
        rhs = parse(groups["rhs"])
    except Exception:
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
    # check for bad options
    check_illegal_options(insn_options, special_insn_kind)

    if special_insn_kind == "gbarrier":
        cls = BarrierInstruction
        kwargs["synchronization_kind"] = "global"
    elif special_insn_kind == "lbarrier":
        cls = BarrierInstruction
        kwargs["synchronization_kind"] = "local"
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
            changed = False

            def checked_intern(s):
                nonlocal changed
                interned_s = intern(s)
                if id(interned_s) != id(s):
                    changed = True
                return interned_s

            def intern_if_str(s):
                if isinstance(s, str):
                    return checked_intern(s)
                else:
                    return s

            copy_args = {
                "id": intern_if_str(insn.id),
                "depends_on": frozenset(intern_if_str(dep)
                    for dep in insn.depends_on),
                "groups": frozenset(checked_intern(grp) for grp in insn.groups),
                "conflicts_with_groups": frozenset(
                    checked_intern(grp) for grp in insn.conflicts_with_groups),
                "within_inames": frozenset(
                    checked_intern(iname) for iname in insn.within_inames),
            }
            if changed:
                insn = insn.copy(**copy_args)
            new_instructions.append(insn)
            continue

        elif not isinstance(insn, str):
            raise TypeError("Instructions must be either an Instruction "
                    "instance or a parseable string. got '%s' instead."
                    % type(insn))

        for sub_insn in insn.split("\n"):
            comment_start = sub_insn.find("#")
            if comment_start >= 0:
                sub_insn = sub_insn[:comment_start]

            sub_insn = sub_insn.strip()
            if not sub_insn:
                continue

            new_instructions.append(sub_insn)

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
            {"predicates": frozenset(),
                "insn_predicates": frozenset()}]

    for insn in instructions:
        if isinstance(insn, InstructionBase):
            local_w_inames = insn_options_stack[-1]["within_inames"]

            if insn.within_inames_is_final:
                if not (local_w_inames <= insn.within_inames):
                    raise LoopyError("non-parsed instruction '%s' without "
                            "inames '%s' (but with final iname dependencies) "
                            "found inside 'for'/'with' block for inames "
                            "'%s'" % (
                                insn.id,
                                ", ".join(local_w_inames - insn.within_inames),
                                local_w_inames))

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
                            insn.conflicts_with_groups
                            | insn_options_stack[-1]["conflicts_with_groups"]),
                        **kwargs)

            norm_tags = _normalize_tags(insn.tags)
            if norm_tags != insn.tags:
                insn = insn.copy(tags=norm_tags)

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
            # check for bad options
            check_illegal_options(insn_options_stack[-1], "with-block")
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
            if_options["insn_predicates"] = options["predicates"]
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
            if obj["predicates"] == if_predicates_stack[-1]["insn_predicates"] and\
                    if_predicates_stack[-1]["insn_predicates"] and\
                    obj["within_inames"] == if_predicates_stack[-1]["within_inames"]:
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
            if subst.name in substitutions:
                raise LoopyError("attempt to redefine substitution rule "
                                 f"'{subst.name}'")
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

    result = {iname.strip() for iname in match.group(1).split(",")
            if iname.strip()}

    return result


EX_QUANT_RE = re.compile(r"\bexists\s+([a-zA-Z0-9])\s*\:")


def _find_existentially_quantified_inames(dom_str):
    return {ex_quant.group(1) for ex_quant in EX_QUANT_RE.finditer(dom_str)}


def parse_domains(domains, defines):
    if isinstance(domains, (isl.BasicSet, str)):
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
                dom = "[{}] -> {}".format(",".join(sorted(parameters)), dom)

            try:
                dom = isl.BasicSet.read_from_str(isl.DEFAULT_CONTEXT, dom)
            except Exception:
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

    if result == []:
        result = [isl.BasicSet("{:}")]

    return result

# }}}


# {{{ guess kernel args (if requested)

class IndexRankFinder(CSECachingMapperMixin, WalkMapper):
    def __init__(self, arg_name):
        self.arg_name = arg_name
        self.index_ranks = []
        WalkMapper.__init__(self)

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
            for pred in insn.predicates:
                self.all_names.update(get_dependencies(self.submap(pred)))

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
        import loopy as lp
        from loopy.kernel.data import ValueArg, ArrayArg

        if arg_name in self.all_params:
            return ValueArg(arg_name)

        if arg_name in self.all_written_names:
            # It's not a temp var, and thereby not a domain parameter--the only
            # other writable type of variable is an argument.

            return ArrayArg(arg_name,
                    shape=lp.auto,
                    offset=self.default_offset,
                    address_space=AddressSpace.GLOBAL)

        irank = self.find_index_rank(arg_name)
        if irank == 0:
            # read-only, no indices
            return ValueArg(arg_name)
        else:
            return ArrayArg(
                    arg_name, shape=lp.auto, offset=self.default_offset,
                    address_space=AddressSpace.GLOBAL)

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

        temp_var_names = set(self.temporary_variables.keys())

        for insn in self.instructions:
            if isinstance(insn, MultiAssignmentBase):
                for assignee_var_name, temp_var_type in zip(
                        insn.assignee_var_names(),
                        insn.temp_var_types):
                    if temp_var_type.has_value:
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
            {arg.name for arg in knl.args}
            | set(knl.temporary_variables.keys()))

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
        super().__init__()

    def map_reduction(self, expr, additional_inames):
        additional_inames = additional_inames | frozenset(expr.inames)

        return super().map_reduction(expr, additional_inames)

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

        if dtype is not None:
            dtype = np.dtype(dtype)

        import loopy as lp
        from loopy.kernel.data import TemporaryVariable
        new_temp_vars.append(TemporaryVariable(
                name=new_var_name,
                dtype=dtype,
                address_space=lp.auto,
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
            new_expression = cseam(insn.expression, frozenset())
            if new_expression is not insn.expression:
                new_insns.append(insn.copy(expression=new_expression))
            else:
                new_insns.append(insn)
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

                if not temp_var_type.has_value:
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
                        dtype=temp_var_type.value,
                        address_space=lp.auto,
                        base_indices=lp.auto,
                        shape=lp.auto,
                        order=default_order)

            if isinstance(insn, Assignment):
                insn = insn.copy(temp_var_type=Optional())
            else:
                insn = insn.copy(temp_var_types=(Optional(),) * len(insn.assignees))

        new_insns.append(insn)

    return knl.copy(
            instructions=new_insns,
            temporary_variables=new_temp_vars)

# }}}


# {{{ determine shapes of temporaries

def find_shapes_of_vars(knl, var_names, feed_expression):
    if not var_names:
        return {}, {}, {}
    from loopy.symbolic import BatchedAccessMapMapper, SubstitutionRuleExpander
    submap = SubstitutionRuleExpander(knl.substitutions)

    armap = BatchedAccessMapMapper(knl, var_names)

    def run_through_armap(expr, inames):
        armap(submap(expr), inames)
        return expr

    feed_expression(run_through_armap)

    var_to_base_indices = {}
    var_to_shape = {}
    var_to_error = {}

    from loopy.diagnostic import StaticValueFindingError

    for var_name in var_names:
        access_range = armap.get_access_range(var_name)
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
    import loopy as lp

    vars_needing_shape_inference = set()
    scalar_vars = set()

    for tv in knl.temporary_variables.values():
        if tv.shape is lp.auto or tv.base_indices is lp.auto:
            vars_needing_shape_inference.add(tv.name)

    from loopy.kernel.instruction import Assignment
    from pymbolic.primitives import Variable
    for insn in knl.instructions:
        # If there's an assignment to a var without a subscript
        # then assume that the variable is a scalar.
        # This is beneficial if afterwards there's no vars
        # needing shape inference
        if isinstance(insn, Assignment) and isinstance(insn.assignee, Variable):
            vars_needing_shape_inference.discard(insn.assignee.name)
            scalar_vars.add(insn.assignee.name)

    def feed_all_expressions(receiver):
        for insn in knl.instructions:
            insn.with_transformed_expressions(
                    lambda expr: receiver(expr, insn.within_inames))  # noqa: B023

    var_to_base_indices, var_to_shape, var_to_error = (
        find_shapes_of_vars(
                knl, vars_needing_shape_inference, feed_all_expressions))

    # {{{ fall back to legacy method

    if len(var_to_error) > 0:
        vars_needing_shape_inference = set(var_to_error.keys())

        for varname, err in var_to_error.items():
            warn_with_kernel(knl, "temp_shape_fallback",
                             "Had to fall back to legacy method of determining "
                             "shape of temporary '%s' because: %s"
                             % (varname, err))

        def feed_assignee_of_instruction(receiver):
            for insn in knl.instructions:
                for assignee in insn.assignees:
                    receiver(assignee, insn.within_inames)

        var_to_base_indices_fallback, var_to_shape_fallback, var_to_error = (
            find_shapes_of_vars(
                    knl, vars_needing_shape_inference, feed_assignee_of_instruction))

        if len(var_to_error) > 0:
            # No way around errors: propagate an exception upward.
            formatted_errors = (
                "\n\n".join("'{}': {}".format(varname, var_to_error[varname])
                for varname in sorted(var_to_error.keys())))

            raise LoopyError("got the following exception(s) trying to find the "
                    "shape of temporary variables: %s" % formatted_errors)

        var_to_base_indices.update(var_to_base_indices_fallback)
        var_to_shape.update(var_to_shape_fallback)

    # }}}

    new_temp_vars = {}
    changed = False

    for tv in knl.temporary_variables.values():
        if tv.name in scalar_vars:
            if tv.base_indices is lp.auto:
                tv = tv.copy(base_indices=())
                changed = True
            if tv.shape is lp.auto:
                tv = tv.copy(shape=())
                changed = True
        else:
            if tv.base_indices is lp.auto:
                tv = tv.copy(base_indices=var_to_base_indices[tv.name])
                changed = True
            if tv.shape is lp.auto:
                tv = tv.copy(shape=var_to_shape[tv.name])
                changed = True
        new_temp_vars[tv.name] = tv

    if not changed:
        return knl

    return knl.copy(temporary_variables=new_temp_vars)

# }}}


# {{{ expand defines in shapes

def expand_defines_in_shapes(kernel, defines):
    if not defines:
        return kernel

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
    for tv in kernel.temporary_variables.values():
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

    var_names = []
    for arg in kernel.args:
        if isinstance(arg, ArrayBase) and arg.shape is lp.auto:
            var_names.append(arg.name)

    if var_names:
        shapes = guess_var_shape(kernel, var_names)
    else:
        shapes = []

    count = 0
    for arg in kernel.args:
        if isinstance(arg, ArrayBase) and arg.shape is lp.auto:
            shape = shapes[count]
            count = count + 1
            arg = arg.copy(shape=shape)

        new_args.append(arg)

    return kernel.copy(args=new_args)

# }}}


# {{{ apply default_order to args

def apply_default_order_to_args(kernel, default_order):
    from loopy.kernel.array import ArrayBase

    processed_args = []
    for arg in kernel.args:
        if isinstance(arg, ArrayBase):
            if default_order in ["c", "f", "C", "F"]:
                if arg.dim_tags is None:
                    arg = arg.copy(order=default_order)
                else:
                    # leave them the way they are
                    pass
            elif default_order is auto:
                if arg.dim_tags is None and arg.shape is not None:
                    assert arg.shape is not auto
                    arg = arg.copy(
                            dim_tags=tuple(
                                FixedStrideArrayDimTag(auto)
                                for i in range(len(arg.shape))))
                    arg = arg.copy(
                            dim_tags=tuple(
                                FixedStrideArrayDimTag(auto)
                                if isinstance(dim_tag, FixedStrideArrayDimTag)
                                else dim_tag
                                for dim_tag in arg.dim_tags))
            else:
                raise ValueError("unexpected value for default_order: "
                                 f"'{default_order}'")

        processed_args.append(arg)

    return kernel.copy(args=processed_args)

# }}}


# {{{ resolve instruction dependencies

WILDCARD_SYMBOLS = "*?["


def _is_wildcard(s):
    return any(c in s for c in WILDCARD_SYMBOLS)


def _resolve_dependencies(what, knl, insn, deps):
    from loopy.transform.instruction import find_instructions
    from loopy.match import MatchExpressionBase

    new_deps = []

    for dep in deps:
        found_any = False

        if isinstance(dep, MatchExpressionBase):
            for new_dep in find_instructions(knl, dep):
                if new_dep.id != insn.id:
                    new_deps.append(new_dep.id)
                    found_any = True
        elif _is_wildcard(dep):
            from fnmatch import fnmatchcase
            for other_insn in knl.instructions:
                if other_insn.id != insn.id and fnmatchcase(other_insn.id, dep):
                    new_deps.append(other_insn.id)
                    found_any = True
        else:
            if dep in knl.id_to_insn:
                new_deps.append(dep)
                found_any = True

        if not found_any and knl.options.check_dep_resolution:
            raise LoopyError("instruction '%s' declared %s on '%s', "
                    "which did not resolve to any instruction present in the "
                    "kernel '%s'. Set the kernel option 'check_dep_resolution'"
                    "to False to disable this check."
                    % (insn.id, what, dep, knl.name))

    for dep_id in new_deps:
        if dep_id not in knl.id_to_insn:
            raise LoopyError("instruction '%s' depends on instruction id '%s', "
                    "which was not found" % (insn.id, dep_id))

    return frozenset(new_deps)


def resolve_dependencies(knl):
    new_insns = []

    for insn in knl.instructions:
        depends_on = _resolve_dependencies(
                "a dependency", knl, insn, insn.depends_on)
        no_sync_with = frozenset(
                (resolved_insn_id, nosync_scope)
                for nosync_dep, nosync_scope in insn.no_sync_with
                for resolved_insn_id in
                _resolve_dependencies("nosync", knl, insn, (nosync_dep,)))

        if depends_on == insn.depends_on and no_sync_with == insn.no_sync_with:
            new_insn = insn
        else:
            new_insn = insn.copy(depends_on=depends_on, no_sync_with=no_sync_with)
        new_insns.append(new_insn)

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

    instructions = []
    for insn in knl.instructions:
        new_within_inames = insn_inames[insn.id]
        if new_within_inames != insn.within_inames:
            insn = insn.copy(within_inames=new_within_inames)
        instructions.append(insn)

    return knl.copy(instructions=instructions)

# }}}


# {{{ apply single-writer heuristic

@for_each_kernel
def apply_single_writer_depencency_heuristic(kernel, warn_if_used=True,
        error_if_used=False):
    logger.debug("%s: default deps" % kernel.name)

    from loopy.transform.subst import expand_subst
    expanded_kernel = expand_subst(kernel)

    writer_map = kernel.writer_map()

    arg_names = {arg.name for arg in kernel.args}

    var_names = arg_names | set(kernel.temporary_variables.keys())

    dep_map = {
            insn.id: insn.read_dependency_names() & var_names
            for insn in expanded_kernel.instructions}

    changed = False
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
                            - {insn.id})

            # }}}

            depends_on = insn.depends_on
            if depends_on is None:
                depends_on = frozenset()

            new_deps = frozenset(auto_deps) | depends_on

            if new_deps != depends_on:
                msg = (
                    "The single-writer dependency heuristic added dependencies "
                    "on instruction ID(s) '%s' to instruction ID '%s' after "
                    "kernel creation is complete. This is deprecated and "
                    "may stop working in the future. "
                    "To fix this, ensure that instruction dependencies "
                    "are added/resolved as soon as possible, ideally at kernel "
                    "creation time."
                    % (", ".join(new_deps - depends_on), insn.id))
                if warn_if_used:
                    warn_with_kernel(kernel, "single_writer_after_creation", msg)
                if error_if_used:
                    raise LoopyError(msg)

                insn = insn.copy(depends_on=new_deps)
                changed = True

        new_insns.append(insn)

    if changed:
        return kernel.copy(instructions=new_insns)
    else:
        return kernel

# }}}


# {{{ slice to sub array ref

def normalize_slice_params(slice, dimension_length):
    """
    Returns the normalized slice parameters ``(start, stop, step)``.

    :arg slice: An instance of :class:`pymbolic.primitives.Slice`.
    :arg dimension_length: Length of the axis being sliced.
    """
    from pymbolic.primitives import Slice
    from numbers import Integral

    assert isinstance(slice, Slice)
    start, stop, step = slice.start, slice.stop, slice.step

    # {{{ defaulting parameters

    if step is None:
        step = 1

    if step == 0:
        raise LoopyError("Slice cannot have 0 step size.")

    if start is None:
        if step > 0:
            start = 0
        else:
            start = dimension_length-1

    if stop is None:
        if step > 0:
            stop = dimension_length
        else:
            stop = -1

    # }}}

    if not isinstance(step, Integral):
        raise LoopyError("Non-integral step sizes lead to non-affine domains =>"
                         " not supported")

    return start, stop, step


class SliceToInameReplacer(IdentityMapper):
    """
    Converts slices to instances of :class:`loopy.symbolic.SubArrayRef`.

    .. attribute:: var_name_gen

        Variable name generator, in order to generate unique inames within the
        kernel domain.

    .. attribute:: knl

        An instance of :class:`loopy.LoopKernel`

    .. attribute:: subarray_ref_bounds

        A :class:`list` (one entry for each :class:`SubArrayRef` to be created)
        of :class:`dict` instances to store the slices enountered in the
        expressions as a mapping from ``iname`` to a tuple of ``(start, stop,
        step)``, which describes the boxy (i.e. affine) constraints imposed on
        the ``iname`` by the corresponding slice notation its intended to
        replace.
    """
    def __init__(self, knl):
        self.subarray_ref_bounds = []
        self.knl = knl
        self.var_name_gen = knl.get_var_name_generator()
        super().__init__()

    def map_subscript(self, expr):
        subscript_iname_bounds = {}

        new_index = []
        swept_inames = []
        for i, index in enumerate(expr.index_tuple):
            if isinstance(index, Slice):
                unique_var_name = self.var_name_gen(based_on="i")
                if expr.aggregate.name in self.knl.arg_dict:
                    shape = self.knl.arg_dict[expr.aggregate.name].shape
                else:
                    assert expr.aggregate.name in self.knl.temporary_variables
                    shape = self.knl.temporary_variables[
                            expr.aggregate.name].shape
                if shape is None or shape[i] is None:
                    raise LoopyError("Slice notation is only supported for "
                            "variables whose shapes are known at creation time "
                            "-- maybe add the shape for '{}'.".format(
                                expr.aggregate.name))

                domain_length = shape[i]
                start, stop, step = normalize_slice_params(index, domain_length)
                subscript_iname_bounds[unique_var_name] = (start, stop, step)
                new_index.append(start+step*Variable(unique_var_name))
                swept_inames.append(Variable(unique_var_name))
            else:
                new_index.append(index)

        if swept_inames:
            self.subarray_ref_bounds.append(subscript_iname_bounds)
            result = SubArrayRef(tuple(swept_inames), Subscript(
                self.rec(expr.aggregate),
                self.rec(tuple(new_index))))
        else:
            result = super().map_subscript(expr)

        return result

    def map_call(self, expr):

        def _convert_array_to_slices(arg):
            # FIXME: We do not support something like A[1] should point to the
            # second row if 'A' is 3 x 3 array.
            if isinstance(arg, Variable):
                from loopy.kernel.data import auto
                if (arg.name in self.knl.temporary_variables):
                    if self.knl.temporary_variables[arg.name].shape in [
                            auto, None]:
                        # do not convert arrays with unknown shapes to slices.
                        # (If an array of unknown shape was passed in error, will be
                        # caught and raised during preprocessing).
                        array_arg_shape = ()
                    else:
                        array_arg_shape = (
                                self.knl.temporary_variables[arg.name].shape)
                elif arg.name in self.knl.arg_dict:
                    if isinstance(self.knl.arg_dict[arg.name], ValueArg):
                        array_arg_shape = ()
                    else:

                        if self.knl.arg_dict[arg.name].shape in [
                                auto, None]:
                            # do not convert arrays with unknown shapes to slices.
                            # (If an array of unknown shape was passed in error, will
                            # be caught and raised during preprocessing).
                            array_arg_shape = ()
                        else:
                            array_arg_shape = (
                                    self.knl.arg_dict[arg.name].shape)
                else:
                    # arg could be either an iname or a "mangled symbol"
                    array_arg_shape = ()

                if array_arg_shape != ():
                    return Subscript(arg, tuple(Slice(())
                                                for _ in array_arg_shape))
            return arg

        return Call(expr.function,
                    tuple(self.rec(_convert_array_to_slices(par))
                          for par in expr.parameters))

    def map_call_with_kwargs(self, expr):
        # See: https://github.com/inducer/loopy/pull/323
        raise NotImplementedError

    def get_iname_domain_as_isl_set(self):
        """
        Returns the extra domain constraints imposed by the slice inames,
        recorded in :attr:`iname_domains`.
        """
        subarray_ref_domains = []
        for sar_bounds in self.subarray_ref_bounds:
            ctx = self.knl.isl_context
            space = isl.Space.create_from_names(ctx,
                    set=list(sar_bounds.keys()))
            from loopy.symbolic import get_dependencies
            args_as_params_for_domains = set()
            for slice_ in sar_bounds.values():
                args_as_params_for_domains |= get_dependencies(slice_)

            space = space.add_dims(dim_type.param, len(args_as_params_for_domains))
            for i, arg in enumerate(args_as_params_for_domains):
                space = space.set_dim_name(dim_type.param, i, arg)

            iname_set = isl.BasicSet.universe(space)

            from loopy.isl_helpers import make_slab
            for iname, (start, stop, step) in sar_bounds.items():
                if step > 0:
                    iname_set = iname_set & make_slab(space, iname, 0,
                                                      stop-start, step)
                else:
                    iname_set = iname_set & make_slab(space, iname, 0,
                                                      start-stop, -step)

            subarray_ref_domains.append(iname_set)

        return subarray_ref_domains


def realize_slices_array_inputs_as_sub_array_refs(kernel):
    """
    Returns a kernel with the instances of :class:`pymbolic.primitives.Slice`
    encountered in expressions replaced as `loopy.symbolic.SubArrayRef`.
    """
    slice_replacer = SliceToInameReplacer(kernel)
    new_insns = [insn.with_transformed_expressions(slice_replacer)
                for insn in kernel.instructions]

    return kernel.copy(
            domains=(
                kernel.domains
                + slice_replacer.get_iname_domain_as_isl_set()),
            instructions=new_insns)

# }}}


# {{{ make_function

def make_function(domains, instructions, kernel_data=None, **kwargs):
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

        A list of :class:`ValueArg`, :class:`ArrayArg`, ... (etc.) instances.
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
    :arg default_order: "C" (default), "F" or :class:`loopy.auto`.
        The default memory layout of arrays that are not explicitly
        specified. If :class:`loopy.auto`, variables for strides are
        automatically created.
    :arg default_offset: 0 or :class:`loopy.auto`. The default value of
        *offset* in :attr:`ArrayArg` for guessed arguments.
        Defaults to 0.
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

    :arg lang_version: The language version against which the kernel was
        written, a tuple. To ensure future compatibility, copy the current value of
        :data:`loopy.MOST_RECENT_LANGUAGE_VERSION` and pass that value.

        (If you just pass :data:`loopy.MOST_RECENT_LANGUAGE_VERSION` directly,
        breaking language changes *will* apply to your kernel without asking,
        likely breaking your code.)

        If not given, this value defaults to version **(2018, 2)** and
        a warning will be issued.

        To set the kernel version for all :mod:`loopy` kernels in a (Python) source
        file, you may simply say::

            from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

        If *lang_version* is not explicitly given, that version value will be used.

        See also :ref:`language-versioning`.

    .. versionchanged:: 2017.2.1

        *lang_version* added.

    .. versionchanged:: 2017.2

        *fixed_parameters* added.

    .. versionchanged:: 2016.3

        *seq_dependencies* added.
    """

    creation_plog = ProcessLogger(
            logger,
            "%s: instantiate" % kwargs.get("name", "(unnamed)"))

    if kernel_data is None:
        kernel_data = [...]
    defines = kwargs.pop("defines", {})
    default_order = kwargs.pop("default_order", "C")
    default_offset = kwargs.pop("default_offset", 0)
    silenced_warnings = kwargs.pop("silenced_warnings", [])
    options = kwargs.pop("options", None)
    flags = kwargs.pop("flags", None)
    target = kwargs.pop("target", None)
    seq_dependencies = kwargs.pop("seq_dependencies", False)
    fixed_parameters = kwargs.pop("fixed_parameters", {})
    assumptions = kwargs.pop("assumptions", None)

    if defines:
        from warnings import warn
        warn("'defines' argument to make_kernel is deprecated. "
                "Use lp.fix_parameters instead",
                DeprecationWarning, stacklevel=2)

    if target is None:
        from loopy import _DEFAULT_TARGET
        target = _DEFAULT_TARGET()

    if flags is not None:
        if options is not None:
            raise TypeError("may not pass both 'options' and 'flags'")

        from warnings import warn
        warn("'flags' is deprecated. Use 'options' instead",
                DeprecationWarning, stacklevel=2)
        options = flags

    from loopy.options import make_options
    options = make_options(options)

    # {{{ handle kernel language version

    from loopy.version import LANGUAGE_VERSION_SYMBOLS

    import loopy.version as v
    version_to_symbol = {
            getattr(v, lvs): lvs
            for lvs in LANGUAGE_VERSION_SYMBOLS}

    lang_version = kwargs.pop("lang_version", None)
    if lang_version is None:
        # {{{ peek into caller's module to look for LOOPY_KERNEL_LANGUAGE_VERSION

        # This *is* gross. But it seems like the right thing interface-wise.
        import inspect
        if inspect.currentframe().f_back.f_code.co_name == "make_kernel":
            # if caller is "make_kernel", read globals from make_kernel's caller
            caller_globals = inspect.currentframe().f_back.f_back.f_globals
        else:
            caller_globals = inspect.currentframe().f_back.f_globals

        for ver_sym in LANGUAGE_VERSION_SYMBOLS:
            try:
                lang_version = caller_globals[ver_sym]
                break
            except KeyError:
                pass

        # }}}

        if lang_version is None:
            from warnings import warn
            from loopy.diagnostic import LoopyWarning
            from loopy.version import (
                    MOST_RECENT_LANGUAGE_VERSION,
                    FALLBACK_LANGUAGE_VERSION)
            warn("'lang_version' was not passed to make_function(). "
                    "To avoid this warning, pass "
                    "lang_version={ver} in this invocation. "
                    "(Or say 'from loopy.version import "
                    "{sym_ver}' in "
                    "the global scope of the calling frame.)"
                    .format(
                        ver=MOST_RECENT_LANGUAGE_VERSION,
                        sym_ver=version_to_symbol[MOST_RECENT_LANGUAGE_VERSION]
                        ),
                    LoopyWarning, stacklevel=2)

            lang_version = FALLBACK_LANGUAGE_VERSION

    if lang_version not in version_to_symbol:
        raise LoopyError(f"Language version '{lang_version}' is not known.")

    # }}}

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

        if isinstance(dat, ArrayBase) and isinstance(dat.shape, tuple):  # noqa pylint:disable=no-member
            new_shape = []
            for shape_axis in dat.shape:  # pylint:disable=no-member
                if shape_axis is not None:
                    new_shape.append(expand_defines_in_expr(shape_axis, defines))
                else:
                    new_shape.append(shape_axis)
            dat = dat.copy(shape=tuple(new_shape))  # pylint:disable=no-member

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

    # {{{ process assumptions

    from loopy.kernel.tools import get_outer_params

    if assumptions is None:
        dom0_space = domains[0].get_space()
        assumptions_space = isl.Space.params_alloc(
                dom0_space.get_ctx(), dom0_space.dim(dim_type.param))
        for i in range(dom0_space.dim(dim_type.param)):
            assumptions_space = assumptions_space.set_dim_name(
                    dim_type.param, i,
                    dom0_space.get_dim_name(dim_type.param, i))
        assumptions = isl.BasicSet.universe(assumptions_space)
    elif isinstance(assumptions, str):
        assumptions_set_str = "[%s] -> { : %s}" \
                % (",".join(s for s in get_outer_params(domains)),
                    assumptions)
        assumptions = isl.BasicSet.read_from_str(domains[0].get_ctx(),
                                                 assumptions_set_str)
    else:
        if not isinstance(assumptions, isl.BasicSet):
            raise LoopyError("assumptions must be either 'str' or BasicSet")

    # }}}

    from loopy.kernel.data import Iname
    from loopy.kernel import _get_inames_from_domains
    inames = {name: Iname(name, frozenset())
              for name in _get_inames_from_domains(domains)}

    arg_guesser = ArgumentGuesser(domains, instructions,
            temporary_variables, substitutions,
            default_offset)

    kernel_args = arg_guesser.convert_names_to_full_args(kernel_args)
    kernel_args = arg_guesser.guess_kernel_args_if_requested(kernel_args)

    for name, rule in kwargs.pop("substitutions", {}).items():
        if name in substitutions:
            raise LoopyError(f"substitution rule '{name}' declared both in-line "
                             "and via substitutions argument")

        substitutions[name] = rule

    kwargs["substitutions"] = substitutions

    from pytools.tag import normalize_tags, check_tag_uniqueness
    tags = check_tag_uniqueness(normalize_tags(kwargs.pop("tags", frozenset())))

    index_dtype = kwargs.pop("index_dtype", None)
    if index_dtype is None:
        index_dtype = np.int32

    from loopy.types import to_loopy_type
    index_dtype = to_loopy_type(index_dtype)

    preambles = kwargs.pop("preambles", None)
    if preambles is None:
        preambles = ()
    elif not isinstance(preambles, tuple):
        preambles = tuple(preambles)

    preamble_generators = kwargs.pop("preamble_generators", None)
    if preamble_generators is None:
        preamble_generators = ()
    elif not isinstance(preamble_generators, tuple):
        preamble_generators = tuple(preamble_generators)

    from loopy.kernel import LoopKernel
    knl = LoopKernel(domains, instructions, kernel_args,
            temporary_variables=temporary_variables,
            silenced_warnings=silenced_warnings,
            options=options,
            target=target,
            tags=tags,
            inames=inames,
            assumptions=assumptions,
            index_dtype=index_dtype,
            preambles=preambles,
            preamble_generators=preamble_generators,
            **kwargs)

    from loopy.transform.instruction import uniquify_instruction_ids
    knl = uniquify_instruction_ids(knl)
    from loopy.check import check_for_duplicate_insn_ids
    check_for_duplicate_insn_ids(knl)

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

    # convert slices to iname domains
    knl = realize_slices_array_inputs_as_sub_array_refs(knl)

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

    from loopy.kernel.tools import infer_args_are_input_output
    knl = infer_args_are_input_output(knl)

    creation_plog.done()

    from loopy.translation_unit import make_program
    return make_program(knl)

# }}}


# {{{ make_kernel

def make_kernel(*args, **kwargs):
    tunit = make_function(*args, **kwargs)
    name, = tunit.callables_table
    return tunit.with_entrypoints(name)


make_kernel.__doc__ = make_function.__doc__

# }}}

# vim: fdm=marker

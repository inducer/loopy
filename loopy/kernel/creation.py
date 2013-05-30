"""UI for kernel creation."""

from __future__ import division

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
from loopy.symbolic import IdentityMapper, WalkMapper
from loopy.kernel.data import Instruction, SubstitutionRule
import islpy as isl
from islpy import dim_type

import re


# {{{ tool: unique name generation

def generate_unique_possibilities(prefix):
    yield prefix

    try_num = 0
    while True:
        yield "%s_%d" % (prefix, try_num)
        try_num += 1


class UniqueNameGenerator:
    def __init__(self, existing_names):
        self.existing_names = existing_names.copy()

    def is_name_conflicting(self, name):
        return name in self.existing_names

    def add_name(self, name):
        if self.is_name_conflicting(name):
            raise ValueError("name '%s' conflicts with existing names")
        self.existing_names.add(name)

    def add_names(self, names):
        for name in names:
            self.add_name(name)

    def __call__(self, based_on="var"):
        for var_name in generate_unique_possibilities(based_on):
            if not self.is_name_conflicting(var_name):
                break

        self.existing_names.add(var_name)
        return var_name

_IDENTIFIER_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")


def _gather_isl_identifiers(s):
    return set(_IDENTIFIER_RE.findall(s)) - set(["and", "or", "exists"])


class MakeUnique:
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

    for find_regexp, replace_pattern in [
            (BRACE_RE, r"\$\{%s\}"),
            (WORD_RE, r"\b%s\b"),
            ]:

        for match in find_regexp.finditer(insn):
            word = match.group(1)

            try:
                value = defines[word]
            except KeyError:
                continue

            if isinstance(value, list):
                if single_valued:
                    raise ValueError("multi-valued macro expansion "
                            "not allowed "
                            "in this context (when expanding '%s')" % word)

                replacements = [
                        rep+((replace_pattern % word, subval),)
                        for rep in replacements
                        for subval in value
                        ]
            else:
                replacements = [
                        rep+((replace_pattern % word, value),)
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

    from loopy.symbolic import SubstitutionMapper
    return SubstitutionMapper(subst_func)(expr)

# }}}

# {{{ parse instructions

INSN_RE = re.compile(
        "\s*(?:\<(?P<temp_var_type>.*?)\>)?"
        "\s*(?P<lhs>.+?)\s*(?<!\:)=\s*(?P<rhs>.+?)"
        "\s*?(?:\{(?P<options>[\s\w=,:]+)\}\s*)?$"
        )
SUBST_RE = re.compile(
        r"^\s*(?P<lhs>.+?)\s*:=\s*(?P<rhs>.+)\s*$"
        )


def parse_insn(insn):
    insn_match = INSN_RE.match(insn)
    subst_match = SUBST_RE.match(insn)
    if insn_match is not None and subst_match is not None:
        raise RuntimeError("instruction parse error: %s" % insn)

    if insn_match is not None:
        groups = insn_match.groupdict()
    elif subst_match is not None:
        groups = subst_match.groupdict()
    else:
        raise RuntimeError("insn parse error")

    from loopy.symbolic import parse
    lhs = parse(groups["lhs"])
    rhs = parse(groups["rhs"])

    if insn_match is not None:
        insn_deps = set()
        insn_id = None
        priority = 0

        if groups["options"] is not None:
            for option in groups["options"].split(","):
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

                if opt_key == "id":
                    insn_id = opt_value
                elif opt_key == "priority":
                    priority = int(opt_value)
                elif opt_key == "dep":
                    insn_deps = set(opt_value.split(":"))
                else:
                    raise ValueError("unrecognized instruction option '%s'"
                            % opt_key)

        if groups["temp_var_type"] is not None:
            if groups["temp_var_type"]:
                temp_var_type = np.dtype(groups["temp_var_type"])
            else:
                import loopy as lp
                temp_var_type = lp.auto
        else:
            temp_var_type = None

        from pymbolic.primitives import Variable, Subscript
        if not isinstance(lhs, (Variable, Subscript)):
            raise RuntimeError("left hand side of assignment '%s' must "
                    "be variable or subscript" % lhs)

        return Instruction(
                    id=insn_id,
                    insn_deps=insn_deps,
                    forced_iname_deps=frozenset(),
                    assignee=lhs, expression=rhs,
                    temp_var_type=temp_var_type,
                    priority=priority)

    elif subst_match is not None:
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


def parse_if_necessary(insn, defines):
    if isinstance(insn, Instruction):
        yield insn
        return
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

        for sub_insn in expand_defines(insn, defines, single_valued=False):
            yield parse_insn(sub_insn)

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


def parse_domains(ctx, domains, defines):
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
                dom = "[%s] -> %s" % (",".join(parameters), dom)

            try:
                dom = isl.BasicSet.read_from_str(ctx, dom)
            except:
                print "failed to parse domain '%s'" % dom
                raise
        else:
            assert isinstance(dom, (isl.Set, isl.BasicSet))
            # assert dom.get_ctx() == ctx

        for i_iname in xrange(dom.dim(dim_type.set)):
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

class IndexRankFinder(WalkMapper):
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


def guess_kernel_args_if_requested(domains, instructions, temporary_variables,
        subst_rules, kernel_args, default_offset):
    # Ellipsis is syntactically allowed in Py3.
    if "..." not in kernel_args and Ellipsis not in kernel_args:
        return kernel_args

    kernel_args = [arg for arg in kernel_args
            if arg is not Ellipsis and arg != "..."]

    from loopy.symbolic import SubstitutionRuleExpander
    submap = SubstitutionRuleExpander(subst_rules)

    # {{{ find names that are *not* arguments

    all_inames = set()
    for dom in domains:
        all_inames.update(dom.get_var_names(dim_type.set))

    temp_var_names = set(temporary_variables.iterkeys())

    for insn in instructions:
        if insn.temp_var_type is not None:
            temp_var_names.add(insn.get_assignee_var_name())

    # }}}

    # {{{ find existing and new arg names

    existing_arg_names = set()
    for arg in kernel_args:
        existing_arg_names.add(arg.name)

    not_new_arg_names = existing_arg_names | temp_var_names | all_inames

    all_names = set()
    all_written_names = set()
    from loopy.symbolic import get_dependencies
    for insn in instructions:
        all_written_names.add(insn.get_assignee_var_name())
        all_names.update(get_dependencies(submap(insn.assignee, insn.id)))
        all_names.update(get_dependencies(submap(insn.expression, insn.id)))

    all_params = set()
    for dom in domains:
        all_params.update(dom.get_var_names(dim_type.param))
    all_params = all_params - all_inames

    new_arg_names = (all_names - not_new_arg_names) | all_params

    # }}}

    def find_index_rank(name):
        irf = IndexRankFinder(name)

        for insn in instructions:
            irf(submap(insn.expression, insn.id))
            irf(submap(insn.assignee, insn.id))

        if not irf.index_ranks:
            return 0
        else:
            from pytools import single_valued
            return single_valued(irf.index_ranks)

    from loopy.kernel.data import ValueArg, GlobalArg
    import loopy as lp
    for arg_name in sorted(new_arg_names):
        if arg_name in all_params:
            kernel_args.append(ValueArg(arg_name))
            continue

        if arg_name in all_written_names:
            # It's not a temp var, and thereby not a domain parameter--the only
            # other writable type of variable is an argument.

            kernel_args.append(
                    GlobalArg(arg_name, shape=lp.auto, offset=default_offset))
            continue

        irank = find_index_rank(arg_name)
        if irank == 0:
            # read-only, no indices
            kernel_args.append(ValueArg(arg_name))
        else:
            kernel_args.append(
                    GlobalArg(arg_name, shape=lp.auto, offset=default_offset))

    return kernel_args

# }}}


# {{{ tag reduction inames as sequential

def tag_reduction_inames_as_sequential(knl):
    result = set()

    def map_reduction(red_expr, rec):
        rec(red_expr.expr)
        result.update(red_expr.inames)

    from loopy.symbolic import ReductionCallbackMapper
    for insn in knl.instructions:
        ReductionCallbackMapper(map_reduction)(insn.expression)

    from loopy.kernel.data import ParallelTag, ForceSequentialTag

    new_iname_to_tag = {}
    for iname in result:
        tag = knl.iname_to_tag.get(iname)
        if tag is not None and isinstance(tag, ParallelTag):
            raise RuntimeError("inconsistency detected: "
                    "reduction iname '%s' has "
                    "a parallel tag" % iname)

        if tag is None:
            new_iname_to_tag[iname] = ForceSequentialTag()

    from loopy import tag_inames
    return tag_inames(knl, new_iname_to_tag)

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
        if not set(insn.forced_iname_deps) <= knl.all_inames():
            raise ValueError("In instruction '%s': "
                    "cannot force dependency on inames '%s'--"
                    "they don't exist" % (
                        insn.id,
                        ",".join(
                            set(insn.forced_iname_deps)-knl.all_inames())))


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
            | set(knl.temporary_variables.iterkeys()))

    for insn in knl.instructions:
        var_name = insn.get_assignee_var_name()

        if var_name not in admissible_vars:
            raise RuntimeError("variable '%s' not declared or not "
                    "allowed for writing" % var_name)

# }}}


# {{{ expand common subexpressions into assignments

class CSEToAssignmentMapper(IdentityMapper):
    def __init__(self, add_assignment):
        self.add_assignment = add_assignment
        self.expr_to_var = {}

    def map_common_subexpression(self, expr):
        try:
            return self.expr_to_var[expr.child]
        except KeyError:
            from loopy.symbolic import TypedCSE
            if isinstance(expr, TypedCSE):
                dtype = expr.dtype
            else:
                dtype = None

            child = self.rec(expr.child)
            from pymbolic.primitives import Variable
            if isinstance(child, Variable):
                return child

            var_name = self.add_assignment(expr.prefix, child, dtype)
            var = Variable(var_name)
            self.expr_to_var[expr.child] = var
            return var


def expand_cses(knl):
    def add_assignment(base_name, expr, dtype):
        if base_name is None:
            base_name = "var"

        new_var_name = var_name_gen(base_name)

        if dtype is None:
            import loopy as lp
            dtype = lp.auto
        else:
            dtype = np.dtype(dtype)

        from loopy.kernel.data import TemporaryVariable
        new_temp_vars[new_var_name] = TemporaryVariable(
                name=new_var_name,
                dtype=dtype,
                is_local=None,
                shape=())

        from pymbolic.primitives import Variable
        insn = Instruction(
                id=knl.make_unique_instruction_id(
                    extra_used_ids=newly_created_insn_ids),
                assignee=Variable(new_var_name), expression=expr)
        newly_created_insn_ids.add(insn.id)
        new_insns.append(insn)

        return new_var_name

    cseam = CSEToAssignmentMapper(add_assignment=add_assignment)

    new_insns = []

    var_name_gen = knl.get_var_name_generator()

    newly_created_insn_ids = set()
    new_temp_vars = knl.temporary_variables.copy()

    for insn in knl.instructions:
        new_insns.append(insn.copy(expression=cseam(insn.expression)))

    return knl.copy(
            instructions=new_insns,
            temporary_variables=new_temp_vars)

# }}}


# {{{ temporary variable creation

def create_temporaries(knl):
    new_insns = []
    new_temp_vars = knl.temporary_variables.copy()

    from loopy.symbolic import AccessRangeMapper

    for insn in knl.instructions:
        from loopy.kernel.data import TemporaryVariable

        if insn.temp_var_type is not None:
            assignee_name = insn.get_assignee_var_name()

            armap = AccessRangeMapper(knl, assignee_name)
            armap(insn.assignee, knl.insn_inames(insn))

            if armap.access_range is not None:
                base_indices, shape = zip(*[
                        knl.cache_manager.base_index_and_length(
                            armap.access_range, i)
                        for i in xrange(armap.access_range.dim(dim_type.set))])
            else:
                base_indices = ()
                shape = ()

            if assignee_name in new_temp_vars:
                raise RuntimeError("cannot create temporary variable '%s'--"
                        "already exists" % assignee_name)
            if assignee_name in knl.arg_dict:
                raise RuntimeError("cannot create temporary variable '%s'--"
                        "already exists as argument" % assignee_name)

            new_temp_vars[assignee_name] = TemporaryVariable(
                    name=assignee_name,
                    dtype=insn.temp_var_type,
                    is_local=None,
                    base_indices=base_indices,
                    shape=shape)

            insn = insn.copy(temp_var_type=None)

        new_insns.append(insn)

    return knl.copy(
            instructions=new_insns,
            temporary_variables=new_temp_vars)

# }}}


# {{{ check for reduction iname duplication

def check_for_reduction_inames_duplication_requests(kernel):

    # {{{ helper function

    def check_reduction_inames(reduction_expr, rec):
        for iname in reduction_expr.inames:
            if iname.startswith("@"):
                raise RuntimeError(
                        "Reduction iname duplication with '@' is no "
                        "longer supported. Use loopy.duplicate_inames "
                        "instead.")

    # }}}

    from loopy.symbolic import ReductionCallbackMapper
    rcm = ReductionCallbackMapper(check_reduction_inames)
    for insn in kernel.instructions:
        rcm(insn.expression)

    for sub_name, sub_rule in kernel.substitutions.iteritems():
        rcm(sub_rule.expression)

# }}}


# {{{ duplicate arguments and expand defines in shapes

def dup_args_and_expand_defines_in_shapes(kernel, defines):
    import loopy as lp
    from loopy.kernel.data import ShapedArg
    from loopy.kernel.creation import expand_defines_in_expr

    processed_args = []
    for arg in kernel.args:
        for arg_name in arg.name.split(","):
            if not arg_name.strip():
                continue

            new_arg = arg.copy(name=arg_name)
            if isinstance(arg, ShapedArg):
                if arg.shape is not None and arg.shape is not lp.auto:
                    new_arg = new_arg.copy(
                            shape=expand_defines_in_expr(arg.shape, defines))
                if arg.strides is not None and arg.strides is not lp.auto:
                    new_arg = new_arg.copy(
                            strides=expand_defines_in_expr(
                                arg.strides, defines))

            processed_args.append(new_arg)

    return kernel.copy(args=processed_args)

# }}}


# {{{ guess argument shapes

def guess_arg_shape_if_requested(kernel, default_order):
    new_args = []

    import loopy as lp
    from loopy.kernel.data import ShapedArg
    from loopy.symbolic import SubstitutionRuleExpander, AccessRangeMapper

    submap = SubstitutionRuleExpander(kernel.substitutions,
            kernel.get_var_name_generator())

    for arg in kernel.args:
        if isinstance(arg, ShapedArg) and (
                arg.shape is lp.auto or arg.strides is lp.auto):
            armap = AccessRangeMapper(kernel, arg.name)

            for insn in kernel.instructions:
                armap(submap(insn.assignee, insn.id),
                        kernel.insn_inames(insn))
                armap(submap(insn.expression, insn.id),
                        kernel.insn_inames(insn))

            if armap.access_range is None:
                # no subscripts found, let's call it a scalar
                shape = ()
            else:
                from loopy.isl_helpers import static_max_of_pw_aff
                from loopy.symbolic import pw_aff_to_expr

                shape = []
                for i in xrange(armap.access_range.dim(dim_type.set)):
                    try:
                        shape.append(
                                pw_aff_to_expr(static_max_of_pw_aff(
                                    kernel.cache_manager.dim_max(
                                        armap.access_range, i) + 1,
                                    constants_only=False)))
                    except:
                        print "While trying to find shape axis %d of "\
                                "argument '%s', the following " \
                                "exception occurred:" % (i, arg.name)
                        raise

                shape = tuple(shape)

            if arg.shape is lp.auto:
                arg = arg.copy(shape=shape)
            if arg.strides is lp.auto:
                from loopy.kernel.data import make_strides
                arg = arg.copy(strides=make_strides(shape, default_order))

        new_args.append(arg)

    return kernel.copy(args=new_args)

# }}}


# {{{ apply default_order to args

def apply_default_order_to_args(kernel, default_order):
    from loopy.kernel.data import ShapedArg

    processed_args = []
    for arg in kernel.args:
        if isinstance(arg, ShapedArg):
            arg = arg.copy(order=default_order)
        processed_args.append(arg)

    return kernel.copy(args=processed_args)

# }}}


# {{{ kernel creation top-level

def make_kernel(device, domains, instructions, kernel_args=["..."], **kwargs):
    """User-facing kernel creation entrypoint.

    :arg device: :class:`pyopencl.Device`
    :arg domains: :class:`islpy.BasicSet`
    :arg instructions:
    :arg kernel_args:

    The following keyword arguments are recognized:

    :arg preambles: a list of (tag, code) tuples that identify preamble
        snippets.
        Each tag's snippet is only included once, at its first occurrence.
        The preambles will be inserted in order of their tags.
    :arg preamble_generators: a list of functions of signature
        (seen_dtypes, seen_functions) where seen_functions is a set of
        (name, c_name, arg_dtypes), generating extra entries for *preambles*.
    :arg defines: a dictionary of replacements to be made in instructions given
        as strings before parsing. A macro instance intended to be replaced
        should look like "MACRO" in the instruction code. The expansion given
        in this parameter is allowed to be a list. In this case, instructions
        are generated for *each* combination of macro values.

        These defines may also be used in the domain and in argument shapes and
        strides. They are expanded only upon kernel creation.
    :arg default_order: "C" (default) or "F"
    :arg default_offset: 0 or :class:`loopy.auto`. The default value of
        *offset* in :attr:`loopy.kernel.data.GlobalArg` for guessed arguments.
        Defaults to 0.
    :arg function_manglers: list of functions of signature (name, arg_dtypes)
        returning a tuple (result_dtype, c_name)
        or a tuple (result_dtype, c_name, arg_dtypes),
        where c_name is the C-level function to be called.
    :arg symbol_manglers: list of functions of signature (name) returning
        a tuple (result_dtype, c_name), where c_name is the C-level symbol to
        be evaluated.
    :arg assumptions: the initial implemented_domain, captures assumptions
        on the parameters. (an isl.Set)
    :arg local_sizes: A dictionary from integers to integers, mapping
        workgroup axes to their sizes, e.g. *{0: 16}* forces axis 0 to be
        length 16.
    :arg temporary_variables:
    """

    defines = kwargs.pop("defines", {})
    default_order = kwargs.pop("default_order", "C")
    default_offset = kwargs.pop("default_offset", 0)

    # {{{ instruction/subst parsing

    parsed_instructions = []
    kwargs["substitutions"] = substitutions = {}

    if isinstance(instructions, str):
        instructions = [instructions]
    for insn in instructions:
        for new_insn in parse_if_necessary(insn, defines):
            if isinstance(new_insn, Instruction):
                parsed_instructions.append(new_insn)
            elif isinstance(new_insn, SubstitutionRule):
                substitutions[new_insn.name] = new_insn
            else:
                raise RuntimeError("unexpected type in instruction parsing")

    instructions = parsed_instructions
    del parsed_instructions

    # }}}

    # {{{ find/create isl_context

    isl_context = None
    for domain in domains:
        if isinstance(domain, isl.BasicSet):
            isl_context = domain.get_ctx()
    if isl_context is None:
        isl_context = isl.Context()
    kwargs["isl_context"] = isl_context

    # }}}

    domains = parse_domains(isl_context, domains, defines)

    kernel_args = guess_kernel_args_if_requested(domains, instructions,
            kwargs.get("temporary_variables", {}), substitutions, kernel_args,
            default_offset)

    from loopy.kernel import LoopKernel
    knl = LoopKernel(device, domains, instructions, kernel_args, **kwargs)

    check_for_nonexistent_iname_deps(knl)
    check_for_reduction_inames_duplication_requests(knl)

    knl = tag_reduction_inames_as_sequential(knl)
    knl = create_temporaries(knl)
    knl = expand_cses(knl)
    knl = dup_args_and_expand_defines_in_shapes(knl, defines)
    knl = guess_arg_shape_if_requested(knl, default_order)
    knl = apply_default_order_to_args(knl, default_order)

    # -------------------------------------------------------------------------
    # Ordering dependency:
    # -------------------------------------------------------------------------
    # Must create temporaries before checking for writes to temporary variables
    # that are domain parameters.
    # -------------------------------------------------------------------------

    check_for_multiple_writes_to_loop_bounds(knl)
    check_for_duplicate_names(knl)
    check_written_variable_names(knl)

    return knl

# }}}

# vim: fdm=marker

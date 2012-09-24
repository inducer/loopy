"""Elements of loopy's user-facing language."""

from __future__ import division

import numpy as np
from pytools import Record, memoize_method
import islpy as isl
from islpy import dim_type

import re




class CannotBranchDomainTree(RuntimeError):
    pass

# {{{ index tags

class IndexTag(Record):
    __slots__ = []

    def __hash__(self):
        raise RuntimeError("use .key to hash index tags")




class ParallelTag(IndexTag):
    pass

class HardwareParallelTag(ParallelTag):
    pass

class UniqueTag(IndexTag):
    @property
    def key(self):
        return type(self)

class AxisTag(UniqueTag):
    __slots__ = ["axis"]

    def __init__(self, axis):
        Record.__init__(self,
                axis=axis)

    @property
    def key(self):
        return (type(self), self.axis)

    def __str__(self):
        return "%s.%d" % (
                self.print_name, self.axis)

class GroupIndexTag(HardwareParallelTag, AxisTag):
    print_name = "g"

class LocalIndexTagBase(HardwareParallelTag):
    pass

class LocalIndexTag(LocalIndexTagBase, AxisTag):
    print_name = "l"

class AutoLocalIndexTagBase(LocalIndexTagBase):
    pass

class AutoFitLocalIndexTag(AutoLocalIndexTagBase):
    def __str__(self):
        return "l.auto"

class IlpBaseTag(ParallelTag):
    pass

class UnrolledIlpTag(IlpBaseTag):
    def __str__(self):
        return "ilp.unr"

class LoopedIlpTag(IlpBaseTag):
    def __str__(self):
        return "ilp.seq"

class UnrollTag(IndexTag):
    def __str__(self):
        return "unr"

class ForceSequentialTag(IndexTag):
    def __str__(self):
        return "forceseq"

def parse_tag(tag):
    if tag is None:
        return tag

    if isinstance(tag, IndexTag):
        return tag

    if not isinstance(tag, str):
        raise ValueError("cannot parse tag: %s" % tag)

    if tag == "for":
        return None
    elif tag in ["unr"]:
        return UnrollTag()
    elif tag in ["ilp", "ilp.unr"]:
        return UnrolledIlpTag()
    elif tag == "ilp.seq":
        return LoopedIlpTag()
    elif tag.startswith("g."):
        return GroupIndexTag(int(tag[2:]))
    elif tag.startswith("l."):
        axis = tag[2:]
        if axis == "auto":
            return AutoFitLocalIndexTag()
        else:
            return LocalIndexTag(int(axis))
    else:
        raise ValueError("cannot parse tag: %s" % tag)

# }}}

# {{{ arguments

class _ShapedArg(Record):
    def __init__(self, name, dtype, shape=None, strides=None, order="C",
            offset=0):
        """
        All of the following are optional. Specify either strides or shape.

        :arg shape:
        :arg strides: like numpy strides, but in multiples of
            data type size
        :arg order:
        :arg offset: Offset from the beginning of the vector from which
            the strides are counted.
        """
        dtype = np.dtype(dtype)

        def parse_if_necessary(x):
            if isinstance(x, str):
                from pymbolic import parse
                return parse(x)
            else:
                return x

        def process_tuple(x):
            x = parse_if_necessary(x)
            if not isinstance(x, tuple):
                x = (x,)

            return tuple(parse_if_necessary(xi) for xi in x)

        if strides is not None:
            strides = process_tuple(strides)

        if shape is not None:
            shape = process_tuple(shape)

        if strides is None and shape is not None:
            from pyopencl.compyte.array import (
                    f_contiguous_strides,
                    c_contiguous_strides)

            if order == "F":
                strides = f_contiguous_strides(1, shape)
            elif order == "C":
                strides = c_contiguous_strides(1, shape)
            else:
                raise ValueError("invalid order: %s" % order)

        Record.__init__(self,
                name=name,
                dtype=dtype,
                strides=strides,
                offset=offset,
                shape=shape)

    @property
    @memoize_method
    def numpy_strides(self):
        return tuple(self.dtype.itemsize*s for s in self.strides)

    @property
    def dimensions(self):
        return len(self.shape)

class GlobalArg(_ShapedArg):
    def __repr__(self):
        return "<GlobalArg '%s' of type %s and shape (%s)>" % (
                self.name, self.dtype, ",".join(str(i) for i in self.shape))

class ArrayArg(GlobalArg):
    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("ArrayArg is a deprecated name of GlobalArg", DeprecationWarning,
                stacklevel=2)
        GlobalArg.__init__(self, *args, **kwargs)

class ConstantArg(_ShapedArg):
    def __repr__(self):
        return "<ConstantArg '%s' of type %s and shape (%s)>" % (
                self.name, self.dtype, ",".join(str(i) for i in self.shape))

class ImageArg(Record):
    def __init__(self, name, dtype, dimensions=None, shape=None):
        dtype = np.dtype(dtype)
        if shape is not None:
            if dimensions is not None and dimensions != len(shape):
                raise RuntimeError("cannot specify both shape and "
                        "disagreeing dimensions in ImageArg")
            dimensions = len(shape)
        else:
            if not isinstance(dimensions, int):
                raise RuntimeError("ImageArg: dimensions must be an integer")

        Record.__init__(self,
                dimensions=dimensions,
                shape=shape,
                dtype=dtype,
                name=name)


    def __repr__(self):
        return "<ImageArg '%s' of type %s>" % (self.name, self.dtype)


class ValueArg(Record):
    def __init__(self, name, dtype, approximately=None):
        Record.__init__(self, name=name, dtype=np.dtype(dtype),
                approximately=approximately)

    def __repr__(self):
        return "<ValueArg '%s' of type %s>" % (self.name, self.dtype)

class ScalarArg(ValueArg):
    def __init__(self, name, dtype, approximately=None):
        from warnings import warn
        warn("ScalarArg is a deprecated name of ValueArg",
                DeprecationWarning, stacklevel=2)

        ValueArg.__init__(self, name, dtype, approximately)

# }}}

# {{{ temporary variable

class TemporaryVariable(Record):
    """
    :ivar name:
    :ivar dtype:
    :ivar shape:
    :ivar storage_shape:
    :ivar base_indices:
    :ivar is_local:
    """

    def __init__(self, name, dtype, shape, is_local, base_indices=None,
            storage_shape=None):
        if base_indices is None:
            base_indices = (0,) * len(shape)

        if shape is not None and not isinstance(shape, tuple):
            shape = tuple(shape)

        Record.__init__(self, name=name, dtype=dtype, shape=shape, is_local=is_local,
                base_indices=base_indices,
                storage_shape=storage_shape)

    @property
    def nbytes(self):
        from pytools import product
        return product(si for si in self.shape)*self.dtype.itemsize

# }}}

# {{{ subsitution rule

class SubstitutionRule(Record):
    """
    :ivar name:
    :ivar arguments:
    :ivar expression:
    """

    def __init__(self, name, arguments, expression):
        Record.__init__(self,
                name=name, arguments=arguments, expression=expression)

    def __str__(self):
        return "%s(%s) := %s" % (
                self.name, ", ".join(self.arguments), self.expression)


# }}}

# {{{ instruction

class Instruction(Record):
    """
    :ivar id: An (otherwise meaningless) identifier that is unique within
        a :class:`LoopKernel`.
    :ivar assignee:
    :ivar expression:
    :ivar forced_iname_deps: a set of inames that are added to the list of iname
        dependencies
    :ivar insn_deps: a list of ids of :class:`Instruction` instances that
        *must* be executed before this one. Note that loop scheduling augments this
        by adding dependencies on any writes to temporaries read by this instruction.
    :ivar boostable: Whether the instruction may safely be executed
        inside more loops than advertised without changing the meaning
        of the program. Allowed values are *None* (for unknown), *True*, and *False*.
    :ivar boostable_into: a set of inames into which the instruction
        may need to be boosted, as a heuristic help for the scheduler.
    :ivar priority: scheduling priority

    The following two instance variables are only used until :func:`loopy.make_kernel` is
    finished:

    :ivar temp_var_type: if not None, a type that will be assigned to the new temporary variable
        created from the assignee
    :ivar duplicate_inames_and_tags: a list of inames used in the instruction that will be duplicated onto
        different inames.
    """
    def __init__(self,
            id, assignee, expression,
            forced_iname_deps=frozenset(), insn_deps=set(), boostable=None,
            boostable_into=None,
            temp_var_type=None, duplicate_inames_and_tags=[],
            priority=0):

        from loopy.symbolic import parse
        if isinstance(assignee, str):
            assignee = parse(assignee)
        if isinstance(expression, str):
            assignee = parse(expression)

        assert isinstance(forced_iname_deps, frozenset)
        assert isinstance(insn_deps, set)

        Record.__init__(self,
                id=id, assignee=assignee, expression=expression,
                forced_iname_deps=forced_iname_deps,
                insn_deps=insn_deps, boostable=boostable,
                boostable_into=boostable_into,
                temp_var_type=temp_var_type,
                duplicate_inames_and_tags=duplicate_inames_and_tags,
                priority=priority)

    @memoize_method
    def reduction_inames(self):
        def map_reduction(expr, rec):
            rec(expr.expr)
            for iname in expr.untagged_inames:
                result.add(iname)

        from loopy.symbolic import ReductionCallbackMapper
        cb_mapper = ReductionCallbackMapper(map_reduction)

        result = set()
        cb_mapper(self.expression)

        return result

    def __str__(self):
        result = "%s: %s <- %s" % (self.id,
                self.assignee, self.expression)

        if self.boostable == True:
            if self.boostable_into:
                result += " (boostable into '%s')" % ",".join(self.boostable_into)
            else:
                result += " (boostable)"
        elif self.boostable == False:
            result += " (not boostable)"
        elif self.boostable is None:
            pass
        else:
            raise RuntimeError("unexpected value for Instruction.boostable")

        options = []

        if self.insn_deps:
            options.append("deps="+":".join(self.insn_deps))
        if self.priority:
            options.append("priority=%d" % self.priority)

        return result

    @memoize_method
    def get_assignee_var_name(self):
        from pymbolic.primitives import Variable, Subscript

        if isinstance(self.assignee, Variable):
            var_name = self.assignee.name
        elif isinstance(self.assignee, Subscript):
            agg = self.assignee.aggregate
            assert isinstance(agg, Variable)
            var_name = agg.name
        else:
            raise RuntimeError("invalid lvalue '%s'" % self.assignee)

        return var_name

    @memoize_method
    def get_assignee_indices(self):
        from pymbolic.primitives import Variable, Subscript

        if isinstance(self.assignee, Variable):
            return ()
        elif isinstance(self.assignee, Subscript):
            result = self.assignee.index
            if not isinstance(result, tuple):
                result = (result,)
            return result
        else:
            raise RuntimeError("invalid lvalue '%s'" % self.assignee)

    @memoize_method
    def get_read_var_names(self):
        from loopy.symbolic import get_dependencies
        return get_dependencies(self.expression)

# }}}

# {{{ expand defines

WORD_RE = re.compile(r"\b([a-zA-Z0-9_]+)\b")

def expand_defines(insn, defines, single_valued=True):
    words = set(match.group(1) for match in WORD_RE.finditer(insn))

    replacements = [()]
    for word in words:
        if word not in defines:
            continue

        value = defines[word]
        if isinstance(value, list):
            if single_valued:
                raise ValueError("multi-valued macro expansion not allowed "
                        "in this context (when expanding '%s')" % word)

            replacements = [
                    rep+((r"\b%s\b" % word, subval),)
                    for rep in replacements
                    for subval in value
                    ]
        else:
            replacements = [
                    rep+((r"\b%s\b" % word, value),)
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

# {{{ function manglers / dtype getters

def default_function_mangler(name, arg_dtypes):
    from loopy.reduction import reduction_function_mangler

    manglers = [reduction_function_mangler]
    for mangler in manglers:
        result = mangler(name, arg_dtypes)
        if result is not None:
            return result

    return None

def opencl_function_mangler(name, arg_dtypes):
    if name == "atan2" and len(arg_dtypes) == 2:
        return arg_dtypes[0], name

    if len(arg_dtypes) == 1:
        arg_dtype, = arg_dtypes

        if arg_dtype.kind == "c":
            if arg_dtype == np.complex64:
                tpname = "cfloat"
            elif arg_dtype == np.complex128:
                tpname = "cdouble"
            else:
                raise RuntimeError("unexpected complex type '%s'" % arg_dtype)

            if name in ["sqrt", "exp", "log",
                    "sin", "cos", "tan",
                    "sinh", "cosh", "tanh"]:
                return arg_dtype, "%s_%s" % (tpname, name)

    if name == "dot":
        scalar_dtype, offset, field_name = arg_dtypes[0].fields["s0"]
        return scalar_dtype, name

    return None

def single_arg_function_mangler(name, arg_dtypes):
    if len(arg_dtypes) == 1:
        dtype, = arg_dtypes
        return dtype, name

    return None

def opencl_symbol_mangler(name):
    # FIXME: should be more picky about exact names
    if name.startswith("FLT_"):
        return np.dtype(np.float32), name
    elif name.startswith("DBL_"):
        return np.dtype(np.float64), name
    elif name.startswith("M_"):
        if name.endswith("_F"):
            return np.dtype(np.float32), name
        else:
            return np.dtype(np.float64), name
    else:
        return None

# }}}

# {{{ preamble generators

def default_preamble_generator(seen_dtypes, seen_functions):
    from loopy.reduction import reduction_preamble_generator

    for result in reduction_preamble_generator(seen_dtypes, seen_functions):
        yield result

    has_double = False
    has_complex = False

    for dtype in seen_dtypes:
        if dtype in [np.float64, np.complex128]:
            has_double = True
        if dtype.kind == "c":
            has_complex = True

    if has_double:
        yield ("00_enable_double", """
            #pragma OPENCL EXTENSION cl_khr_fp64: enable
            """)

    if has_complex:
        if has_double:
            yield ("10_include_complex_header", """
                #define PYOPENCL_DEFINE_CDOUBLE

                #include <pyopencl-complex.h>
                """)
        else:
            yield ("10_include_complex_header", """
                #include <pyopencl-complex.h>
                """)

    c_funcs = set(c_name for name, c_name, arg_dtypes in seen_functions)
    if "int_floor_div" in c_funcs:
        yield ("05_int_floor_div", """
            #define int_floor_div(a,b) \
              (( (a) - \
                 ( ( (a)<0 ) != ( (b)<0 )) \
                  *( (b) + ( (b)<0 ) - ( (b)>=0 ) )) \
               / (b) )
            """)

    if "int_floor_div_pos_b" in c_funcs:
        yield ("05_int_floor_div_pos_b", """
            #define int_floor_div_pos_b(a,b) ( \
                ( (a) - ( ((a)<0) ? ((b)-1) : 0 )  ) / (b) \
                )
            """)


# }}}

# {{{ loop kernel object

def _generate_unique_possibilities(prefix):
    yield prefix

    try_num = 0
    while True:
        yield "%s_%d" % (prefix, try_num)
        try_num += 1

_IDENTIFIER_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")

def _gather_identifiers(s):
    return set(_IDENTIFIER_RE.findall(s))

def _parse_domains(ctx, args_and_vars, domains, defines):
    result = []
    available_parameters = args_and_vars.copy()
    used_inames = set()

    for dom in domains:
        if isinstance(dom, str):
            dom, = expand_defines(dom, defines)

            if not dom.lstrip().startswith("["):
                # i.e. if no parameters are already given
                ids = _gather_identifiers(dom)
                parameters = ids & available_parameters
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
            available_parameters.add(iname)

        result.append(dom)

    return result




class LoopKernel(Record):
    """
    :ivar device: :class:`pyopencl.Device`
    :ivar domains: :class:`islpy.BasicSet`
    :ivar instructions:
    :ivar args:
    :ivar schedule:
    :ivar name:
    :ivar preambles: a list of (tag, code) tuples that identify preamble snippets.
        Each tag's snippet is only included once, at its first occurrence.
        The preambles will be inserted in order of their tags.
    :ivar preamble_generators: a list of functions of signature
        (seen_dtypes, seen_functions) where seen_functions is a set of
        (name, c_name, arg_dtypes), generating extra entries for `preambles`.
    :ivar assumptions: the initial implemented_domain, captures assumptions
        on the parameters. (an isl.Set)
    :ivar local_sizes: A dictionary from integers to integers, mapping
        workgroup axes to their sizes, e.g. *{0: 16}* forces axis 0 to be
        length 16.
    :ivar temporary_variables:
    :ivar iname_to_tag:
    :ivar substitutions: a mapping from substitution names to :class:`SubstitutionRule`
        objects
    :ivar function_manglers: list of functions of signature (name, arg_dtypes)
        returning a tuple (result_dtype, c_name)
        or a tuple (result_dtype, c_name, arg_dtypes),
        where c_name is the C-level function to be called.
    :ivar symbol_manglers: list of functions of signature (name) returning
        a tuple (result_dtype, c_name), where c_name is the C-level symbol to be
        evaluated.
    :ivar defines: a dictionary of replacements to be made in instructions given
        as strings before parsing. A macro instance intended to be replaced should
        look like "MACRO" in the instruction code. The expansion given in this
        parameter is allowed to be a list. In this case, instructions are generated
        for *each* combination of macro values.

        These defines may also be used in the domain and in argument shapes and
        strides. They are expanded only upon kernel creation.

    The following arguments are not user-facing:

    :ivar iname_slab_increments: a dictionary mapping inames to (lower_incr,
        upper_incr) tuples that will be separated out in the execution to generate
        'bulk' slabs with fewer conditionals.
    :ivar applied_iname_rewrites: A list of past substitution dictionaries that
        were applied to the kernel. These are stored so that they may be repeated
        on expressions the user specifies later.
    :ivar cache_manager:
    :ivar isl_context:

    The following instance variables are only used until :func:`loopy.make_kernel` is
    finished:

    :ivar iname_to_tag_requests:
    """

    # {{{ constructor

    def __init__(self, device, domains, instructions, args=[], schedule=None,
            name="loopy_kernel",
            preambles=[],
            preamble_generators=[default_preamble_generator],
            assumptions=None,
            local_sizes={},
            temporary_variables={},
            iname_to_tag={},
            substitutions={},
            function_manglers=[
                default_function_mangler,
                opencl_function_mangler,
                single_arg_function_mangler,
                ],
            symbol_manglers=[opencl_symbol_mangler],
            defines={},

            # non-user-facing
            iname_slab_increments={},
            applied_iname_rewrites=[],
            cache_manager=None,
            iname_to_tag_requests=None,
            index_dtype=np.int32,
            isl_context=None,

            # When kernels get intersected in slab decomposition,
            # their grid sizes shouldn't change. This provides
            # a way to forward sub-kernel grid size requests.
            get_grid_sizes=None):
        """
        :arg domain: a :class:`islpy.BasicSet`, or a string parseable to a basic set by the isl.
            Example: "{[i,j]: 0<=i < 10 and 0<= j < 9}"
        """
        assert not iname_to_tag_requests

        import re

        if cache_manager is None:
            cache_manager = SetOperationCacheManager()

        iname_to_tag_requests = {}

        # {{{ parse instructions

        INAME_ENTRY_RE = re.compile(
                r"^\s*(?P<iname>\w+)\s*(?:\:\s*(?P<tag>[\w.]+))?\s*$")
        INSN_RE = re.compile(
                "\s*(?:\["
                    "(?P<iname_deps_and_tags>[\s\w,:.]*)"
                    "(?:\|(?P<duplicate_inames_and_tags>[\s\w,:.]*))?"
                "\])?"
                "\s*(?:\<(?P<temp_var_type>.*?)\>)?"
                "\s*(?P<lhs>.+?)\s*(?<!\:)=\s*(?P<rhs>.+?)"
                "\s*?(?:\{(?P<options>[\s\w=,:]+)\}\s*)?$"
                )
        SUBST_RE = re.compile(
                r"^\s*(?P<lhs>.+?)\s*:=\s*(?P<rhs>.+)\s*$"
                )

        def parse_iname_and_tag_list(s):
            dup_entries = [
                    dep.strip() for dep in s.split(",")]
            result = []
            for entry in dup_entries:
                if not entry:
                    continue

                entry_match = INAME_ENTRY_RE.match(entry)
                if entry_match is None:
                    raise RuntimeError(
                            "could not parse iname:tag entry '%s'"
                            % entry)

                groups = entry_match.groupdict()
                iname = groups["iname"]
                assert iname

                tag = None
                if groups["tag"] is not None:
                    tag = parse_tag(groups["tag"])

                result.append((iname, tag))

            return result

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
                insn_id = "insn"
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

                if groups["iname_deps_and_tags"] is not None:
                    inames_and_tags = parse_iname_and_tag_list(
                            groups["iname_deps_and_tags"])
                    forced_iname_deps = frozenset(iname for iname, tag in inames_and_tags)
                    iname_to_tag_requests.update(dict(inames_and_tags))
                else:
                    forced_iname_deps = frozenset()

                if groups["duplicate_inames_and_tags"] is not None:
                    duplicate_inames_and_tags = parse_iname_and_tag_list(
                            groups["duplicate_inames_and_tags"])
                else:
                    duplicate_inames_and_tags = []

                if groups["temp_var_type"] is not None:
                    if groups["temp_var_type"]:
                        temp_var_type = np.dtype(groups["temp_var_type"])
                    else:
                        from loopy import infer_type
                        temp_var_type = infer_type
                else:
                    temp_var_type = None

                from pymbolic.primitives import Variable, Subscript
                if not isinstance(lhs, (Variable, Subscript)):
                    raise RuntimeError("left hand side of assignment '%s' must "
                            "be variable or subscript" % lhs)

                parsed_instructions.append(
                        Instruction(
                            id=self.make_unique_instruction_id(
                                parsed_instructions, based_on=insn_id),
                            insn_deps=insn_deps,
                            forced_iname_deps=forced_iname_deps,
                            assignee=lhs, expression=rhs,
                            temp_var_type=temp_var_type,
                            duplicate_inames_and_tags=duplicate_inames_and_tags,
                            priority=priority))

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

                    for arg in lhs.parameters:
                        if not isinstance(arg, Variable):
                            raise RuntimeError("Invalid substitution rule left-hand side")
                        arg_names.append(arg.name)
                else:
                    raise RuntimeError("Invalid substitution rule left-hand side")

                substitutions[subst_name] = SubstitutionRule(
                        name=subst_name,
                        arguments=arg_names,
                        expression=rhs)

        def parse_if_necessary(insn):
            if isinstance(insn, Instruction):
                if insn.id is None:
                    insn = insn.copy(id=self.make_unique_instruction_id(parsed_instructions))
                parsed_instructions.append(insn)
                return

            if not isinstance(insn, str):
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
                    parse_insn(sub_insn)

        parsed_instructions = []

        substitutions = substitutions.copy()

        if isinstance(instructions, str):
            instructions = [instructions]
        for insn in instructions:
            # must construct list one-by-one to facilitate unique id generation
            parse_if_necessary(insn)

        if len(set(insn.id for insn in parsed_instructions)) != len(parsed_instructions):
            raise RuntimeError("instruction ids do not appear to be unique")

        # }}}

        # Ordering dependency:
        # Domain construction needs to know what temporary variables are
        # available. That information can only be obtained once instructions
        # are parsed.

        # {{{ construct domains

        if isinstance(domains, str):
            domains = [domains]

        for domain in domains:
            if isinstance(domain, isl.BasicSet):
                isl_context = domain.get_ctx()
        if isl_context is None:
            isl_context = isl.Context()

        scalar_arg_names = set(arg.name for arg in args if isinstance(arg, ValueArg))
        var_names = (
                set(temporary_variables)
                | set(insn.get_assignee_var_name()
                    for insn in parsed_instructions
                    if insn.temp_var_type is not None))
        domains = _parse_domains(isl_context, scalar_arg_names | var_names, domains,
                defines)

        # }}}

        # {{{ process assumptions

        if assumptions is None:
            dom0_space = domains[0].get_space()
            assumptions_space = isl.Space.params_alloc(
                    dom0_space.get_ctx(), dom0_space.dim(dim_type.param))
            for i in xrange(dom0_space.dim(dim_type.param)):
                assumptions_space = assumptions_space.set_dim_name(
                        dim_type.param, i, dom0_space.get_dim_name(dim_type.param, i))
            assumptions = isl.BasicSet.universe(assumptions_space)

        elif isinstance(assumptions, str):
            all_inames = set()
            all_params = set()
            for dom in domains:
                all_inames.update(dom.get_var_names(dim_type.set))
                all_params.update(dom.get_var_names(dim_type.param))

            domain_parameters = all_params-all_inames

            assumptions_set_str = "[%s] -> { : %s}" \
                    % (",".join(s for s in domain_parameters),
                        assumptions)
            assumptions = isl.BasicSet.read_from_str(domains[0].get_ctx(),
                    assumptions_set_str)

        assert assumptions.is_params()

        # }}}

        # {{{ expand macros in arg shapes

        processed_args = []
        for arg in args:
            for name in arg.name.split(","):
                new_arg = arg.copy(name=name)
                if isinstance(arg, _ShapedArg):
                    if arg.shape is not None:
                        new_arg = new_arg.copy(shape=expand_defines_in_expr(arg.shape, defines))
                    if arg.strides is not None:
                        new_arg = new_arg.copy(strides=expand_defines_in_expr(arg.strides, defines))

                processed_args.append(new_arg)

        # }}}

        index_dtype = np.dtype(index_dtype)
        if index_dtype.kind != 'i':
            raise TypeError("index_dtype must be an integer")
        if np.iinfo(index_dtype).min >= 0:
            raise TypeError("index_dtype must be signed")

        if get_grid_sizes is not None:
            # overwrites method down below
            self.get_grid_sizes = get_grid_sizes

        Record.__init__(self,
                device=device, domains=domains,
                instructions=parsed_instructions,
                args=processed_args,
                schedule=schedule,
                name=name,
                preambles=preambles,
                preamble_generators=preamble_generators,
                assumptions=assumptions,
                iname_slab_increments=iname_slab_increments,
                temporary_variables=temporary_variables,
                local_sizes=local_sizes,
                iname_to_tag=iname_to_tag,
                iname_to_tag_requests=iname_to_tag_requests,
                substitutions=substitutions,
                cache_manager=cache_manager,
                applied_iname_rewrites=applied_iname_rewrites,
                function_manglers=function_manglers,
                symbol_manglers=symbol_manglers,
                index_dtype=index_dtype,
                isl_context=isl_context)

    # }}}

    # {{{ function mangling

    def register_function_mangler(self, mangler):
        return self.copy(
                function_manglers=[mangler]+self.function_manglers)

    def mangle_function(self, identifier, arg_dtypes):
        for mangler in self.function_manglers:
            mangle_result = mangler(identifier, arg_dtypes)
            if mangle_result is not None:
                return mangle_result

        return None

    # }}}

    # {{{ name wrangling

    @memoize_method
    def non_iname_variable_names(self):
        return (set(self.arg_dict.iterkeys())
                | set(self.temporary_variables.iterkeys()))

    @memoize_method
    def all_variable_names(self):
        return (
                set(self.temporary_variables.iterkeys())
                | set(self.substitutions.iterkeys())
                | set(arg.name for arg in self.args)
                | set(self.all_inames()))

    def make_unique_var_name(self, based_on="var", extra_used_vars=set()):
        used_vars = self.all_variable_names() | extra_used_vars

        for var_name in _generate_unique_possibilities(based_on):
            if var_name not in used_vars:
                return var_name

    def make_unique_instruction_id(self, insns=None, based_on="insn", extra_used_ids=set()):
        if insns is None:
            insns = self.instructions

        used_ids = set(insn.id for insn in insns) | extra_used_ids

        for id_str in _generate_unique_possibilities(based_on):
            if id_str not in used_ids:
                return id_str

    def get_var_descriptor(self, name):
        try:
            return self.arg_dict[name]
        except KeyError:
            pass

        try:
            return self.temporary_variables[name]
        except KeyError:
            pass

        raise ValueError("nothing known about variable '%s'" % name)

    @property
    @memoize_method
    def id_to_insn(self):
        return dict((insn.id, insn) for insn in self.instructions)

    # }}}

    # {{{ domain wrangling

    @memoize_method
    def parents_per_domain(self):
        """Return a list corresponding to self.domains (by index)
        containing domain indices which are nested around this
        domain.

        Each domains nest list walks from the leaves of the nesting
        tree to the root.
        """

        # The stack of iname sets records which inames are active
        # as we step through the linear list of domains. It also
        # determines the granularity of inames to be popped/decactivated
        # if we ascend a level.

        iname_set_stack = []
        result = []

        writer_map = self.writer_map()

        for dom in self.domains:
            parameters = set(dom.get_var_names(dim_type.param))
            inames = set(dom.get_var_names(dim_type.set))

            # This next domain may be nested inside the previous domain.
            # Or it may not, in which case we need to figure out how many
            # levels of parents we need to discard in order to find the
            # true parent.

            discard_level_count = 0
            while discard_level_count < len(iname_set_stack):
                # {{{ check for parenthood by loop bound iname

                last_inames = iname_set_stack[-1-discard_level_count]
                if last_inames & parameters:
                    break

                # }}}

                # {{{ check for parenthood by written variable

                is_parent_by_variable = False
                for par in parameters:
                    if par in self.temporary_variables:
                        writer_insns = writer_map[par]

                        if len(writer_insns) > 1:
                            raise RuntimeError("loop bound '%s' "
                                    "may only be written to once" % par)

                        writer_insn, = writer_insns
                        writer_inames = self.insn_inames(writer_insn)

                        if writer_inames & last_inames:
                            is_parent_by_variable = True
                            break

                if is_parent_by_variable:
                    break

                # }}}

                discard_level_count += 1

            if discard_level_count:
                iname_set_stack = iname_set_stack[:-discard_level_count]

            if result:
                parent = len(result)-1
            else:
                parent = None

            for i in range(discard_level_count):
                assert parent is not None
                parent = result[parent]

            # found this domain's parent
            result.append(parent)

            if iname_set_stack:
                parent_inames = iname_set_stack[-1]
            else:
                parent_inames = set()
            iname_set_stack.append(parent_inames | inames)

        return result

    @memoize_method
    def all_parents_per_domain(self):
        """Return a list corresponding to self.domains (by index)
        containing domain indices which are nested around this
        domain.

        Each domains nest list walks from the leaves of the nesting
        tree to the root.
        """
        result = []

        ppd = self.parents_per_domain()
        for dom, parent in zip(self.domains, ppd):
            # keep walking up tree to find *all* parents
            dom_result = []
            while parent is not None:
                dom_result.insert(0, parent)
                parent = ppd[parent]

            result.append(dom_result)

        return result

    @memoize_method
    def _get_home_domain_map(self):
        return dict(
                (iname, i_domain)
                for i_domain, dom in enumerate(self.domains)
                for iname in dom.get_var_names(dim_type.set))

    def get_home_domain_index(self, iname):
        return self._get_home_domain_map()[iname]

    @memoize_method
    def combine_domains(self, domains):
        """
        :arg domains: domain indices of domains to be combined. More 'dominant'
            domains (those which get most say on the actual dim_type of an iname)
            must be later in the order.
        """
        assert isinstance(domains, tuple) # for caching

        if not domains:
            return isl.BasicSet.universe(isl.Space.set_alloc(
                self.isl_context, 0, 0))

        result = None
        for dom_index in domains:
            dom = self.domains[dom_index]
            if result is None:
                result = dom
            else:
                aligned_dom, aligned_result = isl.align_two(
                        dom, result, across_dim_types=True)
                result = aligned_result & aligned_dom

        return result

    def get_inames_domain(self, inames):
        if not inames:
            return self.combine_domains(())

        if isinstance(inames, str):
            inames = frozenset([inames])
        if not isinstance(inames, frozenset):
            inames = frozenset(inames)

            from warnings import warn
            warn("get_inames_domain did not get a frozenset", stacklevel=2)

        return self._get_inames_domain_backend(inames)

    @memoize_method
    def get_leaf_domain_index(self, inames):
        """Find the leaf of the domain tree needed to cover all inames."""

        hdm = self._get_home_domain_map()
        ppd = self.all_parents_per_domain()

        domain_indices = set()

        leaf_domain_index = None

        for iname in inames:
            home_domain_index = hdm[iname]
            if home_domain_index in domain_indices:
                # nothin' new
                continue

            leaf_domain_index = home_domain_index

            all_parents = set(ppd[home_domain_index])
            if not domain_indices <= all_parents:
                raise CannotBranchDomainTree("iname set '%s' requires "
                        "branch in domain tree (when adding '%s')"
                        % (", ".join(inames), iname))

            domain_indices.add(home_domain_index)
            domain_indices.update(all_parents)

        return leaf_domain_index

    @memoize_method
    def _get_inames_domain_backend(self, inames):
        leaf_dom_idx = self.get_leaf_domain_index(inames)

        return self.combine_domains(tuple(sorted(
            self.all_parents_per_domain()[leaf_dom_idx]
            + [leaf_dom_idx]
            )))

    # }}}

    # {{{ iname wrangling

    @memoize_method
    def all_inames(self):
        result = set()
        for dom in self.domains:
            result.update(dom.get_var_names(dim_type.set))
        return frozenset(result)

    @memoize_method
    def all_insn_inames(self):
        """Return a mapping from instruction ids to inames inside which
        they should be run.
        """

        return find_all_insn_inames(self)

    @memoize_method
    def all_referenced_inames(self):
        result = set()
        for inames in self.all_insn_inames().itervalues():
            result.update(inames)
        return result

    def insn_inames(self, insn):
        if isinstance(insn, Instruction):
            return self.all_insn_inames()[insn.id]
        else:
            return self.all_insn_inames()[insn]

    @memoize_method
    def iname_to_insns(self):
        result = dict(
                (iname, set()) for iname in self.all_inames())
        for insn in self.instructions:
            for iname in self.insn_inames(insn):
                result[iname].add(insn.id)

        return result

    # }}}

    # {{{ read and written variables

    @memoize_method
    def reader_map(self):
        """
        :return: a dict that maps variable names to ids of insns that read that variable.
        """
        result = {}

        admissible_vars = (
                set(arg.name for arg in self.args)
                | set(self.temporary_variables.iterkeys()))

        for insn in self.instructions:
            for var_name in insn.get_read_var_names() & admissible_vars:
                result.setdefault(var_name, set()).add(insn.id)

    @memoize_method
    def writer_map(self):
        """
        :return: a dict that maps variable names to ids of insns that write to that variable.
        """
        result = {}

        for insn in self.instructions:
            var_name = insn.get_assignee_var_name()
            var_names = [var_name]

            for var_name in var_names:
                result.setdefault(var_name, set()).add(insn.id)

        return result

    @memoize_method
    def get_read_variables(self):
        result = set()
        for insn in self.instructions:
            result.update(insn.get_read_var_names())
        return result

    @memoize_method
    def get_written_variables(self):
        return frozenset(
            insn.get_assignee_var_name()
            for insn in self.instructions)

    # }}}

    # {{{ argument wrangling

    @property
    @memoize_method
    def arg_dict(self):
        return dict((arg.name, arg) for arg in self.args)

    @property
    @memoize_method
    def scalar_loop_args(self):
        if self.args is None:
            return []
        else:
            from pytools import flatten
            loop_arg_names = list(flatten(dom.get_var_names(dim_type.param)
                    for dom in self.domains))
            return [arg.name for arg in self.args if isinstance(arg, ValueArg)
                    if arg.name in loop_arg_names]
    # }}}

    # {{{ bounds finding

    @memoize_method
    def get_iname_bounds(self, iname):
        domain = self.get_inames_domain(frozenset([iname]))
        d_var_dict = domain.get_var_dict()

        assumptions, domain = isl.align_two(self.assumptions, domain)

        dom_intersect_assumptions = assumptions & domain

        lower_bound_pw_aff = (
                self.cache_manager.dim_min(
                    dom_intersect_assumptions,
                    d_var_dict[iname][1])
                .coalesce())
        upper_bound_pw_aff = (
                self.cache_manager.dim_max(
                    dom_intersect_assumptions,
                    d_var_dict[iname][1])
                .coalesce())

        class BoundsRecord(Record):
            pass

        size = (upper_bound_pw_aff - lower_bound_pw_aff + 1)
        size = size.gist(self.assumptions)

        return BoundsRecord(
                lower_bound_pw_aff=lower_bound_pw_aff,
                upper_bound_pw_aff=upper_bound_pw_aff,
                size=size)

    def find_var_base_indices_and_shape_from_inames(
            self, inames, cache_manager, context=None):
        if not inames:
            return [], []

        base_indices_and_sizes = [
                cache_manager.base_index_and_length(
                    self.get_inames_domain(iname), iname, context)
                for iname in inames]
        return zip(*base_indices_and_sizes)

    @memoize_method
    def get_constant_iname_length(self, iname):
        from loopy.isl_helpers import static_max_of_pw_aff
        from loopy.symbolic import aff_to_expr
        return int(aff_to_expr(static_max_of_pw_aff(
                self.get_iname_bounds(iname).size,
                constants_only=True)))

    @memoize_method
    def get_grid_sizes(self, ignore_auto=False):
        all_inames_by_insns = set()
        for insn in self.instructions:
            all_inames_by_insns |= self.insn_inames(insn)

        if not all_inames_by_insns <= self.all_inames():
            raise RuntimeError("some inames collected from instructions (%s) "
                    "are not present in domain (%s)"
                    % (", ".join(sorted(all_inames_by_insns)),
                        ", ".join(sorted(self.all_inames()))))

        global_sizes = {}
        local_sizes = {}

        from loopy.kernel import (
                GroupIndexTag, LocalIndexTag,
                AutoLocalIndexTagBase)

        for iname in self.all_inames():
            tag = self.iname_to_tag.get(iname)

            if isinstance(tag, GroupIndexTag):
                tgt_dict = global_sizes
            elif isinstance(tag, LocalIndexTag):
                tgt_dict = local_sizes
            elif isinstance(tag, AutoLocalIndexTagBase) and not ignore_auto:
                raise RuntimeError("cannot find grid sizes if automatic local index tags are "
                        "present")
            else:
                tgt_dict = None

            if tgt_dict is None:
                continue

            size = self.get_iname_bounds(iname).size

            if tag.axis in tgt_dict:
                size = tgt_dict[tag.axis].max(size)

            from loopy.isl_helpers import static_max_of_pw_aff
            try:
                # insist block size is constant
                size = static_max_of_pw_aff(size,
                        constants_only=isinstance(tag, LocalIndexTag))
            except ValueError:
                pass

            tgt_dict[tag.axis] = size

        max_dims = self.device.max_work_item_dimensions

        def to_dim_tuple(size_dict, which, forced_sizes={}):
            forced_sizes = forced_sizes.copy()

            size_list = []
            sorted_axes = sorted(size_dict.iterkeys())

            while sorted_axes or forced_sizes:
                if sorted_axes:
                    cur_axis = sorted_axes.pop(0)
                else:
                    cur_axis = None

                if len(size_list) in forced_sizes:
                    size_list.append(
                           forced_sizes.pop(len(size_list)))
                    continue

                assert cur_axis is not None

                if cur_axis > len(size_list):
                    raise RuntimeError("%s axis %d unused" % (
                        which, len(size_list)))

                size_list.append(size_dict[cur_axis])

            if len(size_list) > max_dims:
                raise ValueError("more %s dimensions assigned than supported "
                        "by hardware (%d > %d)" % (which, len(size_list), max_dims))

            return tuple(size_list)

        return (to_dim_tuple(global_sizes, "global"),
                to_dim_tuple(local_sizes, "local", forced_sizes=self.local_sizes))

    def get_grid_sizes_as_exprs(self, ignore_auto=False):
        grid_size, group_size = self.get_grid_sizes(ignore_auto=ignore_auto)

        def tup_to_exprs(tup):
            from loopy.symbolic import pw_aff_to_expr
            return tuple(pw_aff_to_expr(i, int_ok=True) for i in tup)

        return tup_to_exprs(grid_size), tup_to_exprs(group_size)

    # }}}

    # {{{ local memory

    @memoize_method
    def local_var_names(self):
        return set(
                tv.name
            for tv in self.temporary_variables.itervalues()
            if tv.is_local)

    def local_mem_use(self):
        return sum(lv.nbytes for lv in self.temporary_variables.itervalues()
                if lv.is_local)

    # }}}

    def map_expressions(self, func, exclude_instructions=False):
        if exclude_instructions:
            new_insns = self.instructions
        else:
            new_insns = [insn.copy(
                expression=func(insn.expression),
                assignee=func(insn.assignee),
                )
                    for insn in self.instructions]

        return self.copy(
                instructions=new_insns,
                substitutions=dict(
                    (subst.name, subst.copy(expression=func(subst.expression)))
                    for subst in self.substitutions.itervalues()))

    # {{{ pretty-printing

    def __str__(self):
        lines = []

        sep = 75*"-"
        lines.append(sep)
        lines.append("INAME-TO-TAG MAP:")
        for iname in sorted(self.all_inames()):
            line = "%s: %s" % (iname, self.iname_to_tag.get(iname))
            lines.append(line)

        lines.append(sep)
        lines.append("DOMAINS:")
        for dom, parents in zip(self.domains, self.all_parents_per_domain()):
            lines.append(len(parents)*"  " + str(dom))

        if self.substitutions:
            lines.append(sep)
            lines.append("SUBSTIUTION RULES:")
            for rule in self.substitutions.itervalues():
                lines.append(str(rule))

        lines.append(sep)
        lines.append("INSTRUCTIONS:")
        loop_list_width = 35
        for insn in self.instructions:
            loop_list = ",".join(sorted(self.insn_inames(insn)))

            options = [insn.id]
            if insn.priority:
                options.append("priority=%d" % insn.priority)

            if len(loop_list) > loop_list_width:
                lines.append("[%s]" % loop_list)
                lines.append("%s%s <- %s   # %s" % (
                    (loop_list_width+2)*" ", insn.assignee,
                    insn.expression, ", ".join(options)))
            else:
                lines.append("[%s]%s%s <- %s   # %s" % (
                    loop_list, " "*(loop_list_width-len(loop_list)),
                    insn.assignee, insn.expression, ", ".join(options)))

        lines.append(sep)
        lines.append("DEPENDENCIES:")
        for insn in self.instructions:
            if insn.insn_deps:
                lines.append("%s : %s" % (insn.id, ",".join(insn.insn_deps)))
        lines.append(sep)

        if self.schedule is not None:
            lines.append("SCHEDULE:")
            from loopy.schedule import dump_schedule
            lines.append(dump_schedule(self.schedule))
            lines.append(sep)

        return "\n".join(lines)

    # }}}

# }}}

# {{{ find_all_insn_inames fixed point iteration

def find_all_insn_inames(kernel):
    from loopy.symbolic import get_dependencies

    writer_map = kernel.writer_map()

    insn_id_to_inames = {}
    insn_assignee_inames = {}

    for insn in kernel.instructions:
        read_deps = get_dependencies(insn.expression)
        write_deps = get_dependencies(insn.assignee)
        deps = read_deps | write_deps

        iname_deps = (
                deps & kernel.all_inames()
                | insn.forced_iname_deps)

        insn_id_to_inames[insn.id] = iname_deps
        insn_assignee_inames[insn.id] = write_deps & kernel.all_inames()

    temp_var_names = set(kernel.temporary_variables.iterkeys())

    # fixed point iteration until all iname dep sets have converged

    # Why is fixed point iteration necessary here? Consider the following
    # scenario:
    #
    # z = expr(iname)
    # y = expr(z)
    # x = expr(y)
    #
    # x clearly has a dependency on iname, but this is not found until that
    # dependency has propagated all the way up. Doing this recursively is
    # not guaranteed to terminate because of circular dependencies.

    while True:
        did_something = False
        for insn in kernel.instructions:

            # {{{ depdency-based propagation

            # For all variables that insn depends on, find the intersection
            # of iname deps of all writers, and add those to insn's
            # dependencies.

            for tv_name in (get_dependencies(insn.expression)
                    & temp_var_names):
                implicit_inames = None

                for writer_id in writer_map[tv_name]:
                    writer_implicit_inames = (
                            insn_id_to_inames[writer_id]
                            - insn_assignee_inames[writer_id])
                    if implicit_inames is None:
                        implicit_inames = writer_implicit_inames
                    else:
                        implicit_inames = (implicit_inames
                                & writer_implicit_inames)

                inames_old = insn_id_to_inames[insn.id]
                inames_new = (inames_old | implicit_inames) \
                            - insn.reduction_inames()
                insn_id_to_inames[insn.id] = inames_new

                if inames_new != inames_old:
                    did_something = True

            # }}}

            # {{{ domain-based propagation

            # Add all inames occurring in parameters of domains that my current
            # inames refer to.

            inames_old = insn_id_to_inames[insn.id]
            inames_new = set(insn_id_to_inames[insn.id])

            for iname in inames_old:
                home_domain = kernel.domains[kernel.get_home_domain_index(iname)]

                for par in home_domain.get_var_names(dim_type.param):
                    if par in kernel.all_inames():
                        inames_new.add(par)

            if inames_new != inames_old:
                did_something = True
                insn_id_to_inames[insn.id] = frozenset(inames_new)

            # }}}

        if not did_something:
            break

    return insn_id_to_inames

# }}}

# {{{ set operation cache

class SetOperationCacheManager:
    def __init__(self):
        # mapping: set hash -> [(set, op, args, result)]
        self.cache = {}

    def op(self, set, op_name, op, args):
        hashval = hash(set)
        bucket = self.cache.setdefault(hashval, [])

        for bkt_set, bkt_op, bkt_args, result  in bucket:
            if set.plain_is_equal(bkt_set) and op == bkt_op and args == bkt_args:
                return result

        #print op, set.get_dim_name(dim_type.set, args[0])
        result = op(*args)
        bucket.append((set, op_name, args, result))
        return result

    def dim_min(self, set, *args):
        return self.op(set, "dim_min", set.dim_min, args)

    def dim_max(self, set, *args):
        return self.op(set, "dim_max", set.dim_max, args)

    def base_index_and_length(self, set, iname, context=None):
        iname_to_dim = set.space.get_var_dict()
        lower_bound_pw_aff = self.dim_min(set, iname_to_dim[iname][1])
        upper_bound_pw_aff = self.dim_max(set, iname_to_dim[iname][1])

        from loopy.isl_helpers import static_max_of_pw_aff, static_value_of_pw_aff
        from loopy.symbolic import pw_aff_to_expr

        size = pw_aff_to_expr(static_max_of_pw_aff(
                upper_bound_pw_aff - lower_bound_pw_aff + 1, constants_only=True,
                context=context))
        base_index = pw_aff_to_expr(
            static_value_of_pw_aff(lower_bound_pw_aff, constants_only=False,
                context=context))

        return base_index, size

# }}}

# {{{ domain change helper

class DomainChanger:
    """Helps change the domain responsible for *inames* within a kernel.

    .. note: Does not perform an in-place change!
    """

    def __init__(self, kernel, inames):
        self.kernel = kernel
        if inames:
            self.leaf_domain_index = kernel.get_leaf_domain_index(inames)
            self.domain = kernel.domains[self.leaf_domain_index]

        else:
            self.domain = kernel.combine_domains(())
            self.leaf_domain_index = None

    def get_domains_with(self, replacement):
        result = self.kernel.domains[:]
        if self.leaf_domain_index is not None:
            result[self.leaf_domain_index] = replacement
        else:
            result.append(replacement)

        return result

# }}}


# {{{ dot export

def get_dot_dependency_graph(kernel, iname_cluster=False, iname_edge=True):
    lines = []
    for insn in kernel.instructions:
        lines.append("%s [shape=\"box\"];" % insn.id)
        for dep in insn.insn_deps:
            lines.append("%s -> %s;" % (dep, insn.id))

        if iname_edge:
            for iname in kernel.insn_inames(insn):
                lines.append("%s -> %s [style=\"dotted\"];" % (iname, insn.id))

    if iname_cluster:
        for iname in kernel.all_inames():
            lines.append("subgraph cluster_%s { label=\"%s\" %s }" % (iname, iname,
                " ".join(insn.id for insn in kernel.instructions
                    if iname in kernel.insn_inames(insn))))

    return "digraph loopy_deps {\n%s\n}" % "\n".join(lines)

# }}}

# vim: foldmethod=marker

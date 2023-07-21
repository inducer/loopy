__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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

import re

from sys import intern
from immutables import Map

import loopy as lp
import numpy as np
from warnings import warn
from loopy.frontend.fortran.tree import FTreeWalkerBase
from loopy.diagnostic import warn_with_kernel
from loopy.frontend.fortran.diagnostic import (
        TranslationError, TranslatorWarning)
import islpy as isl
from islpy import dim_type
from loopy.symbolic import (IdentityMapper, RuleAwareIdentityMapper,
        SubstitutionRuleMappingContext)
from loopy.diagnostic import LoopyError
from loopy.kernel.instruction import LegacyStringInstructionTag
from pymbolic.primitives import (Wildcard, Slice)


# {{{ subscript base shifter

class SubscriptIndexAdjuster(IdentityMapper):
    """Adjust base indices of subscripts and lengths of slices."""

    def __init__(self, scope):
        self.scope = scope
        super().__init__()

    def get_cache_key(self, expr):
        return super().get_cache_key(expr) + (self.scope,)

    def map_subscript(self, expr):
        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        name = expr.aggregate.name
        dims = self.scope.dim_map.get(name)
        if dims is None:
            return IdentityMapper.map_subscript(self, expr)

        subscript = expr.index

        if not isinstance(subscript, tuple):
            subscript = (subscript,)

        if len(dims) != len(subscript):
            raise TranslationError("inconsistent number of indices "
                    "to '%s'" % name)

        new_subscript = []
        for i in range(len(dims)):
            if len(dims[i]) == 2:
                # has an explicit base index
                base_index, end_index = dims[i]
            elif len(dims[i]) == 1:
                base_index = 1
                end_index, = dims[i]

            sub_i = subscript[i]
            if isinstance(sub_i, Slice):
                start = sub_i.start
                if start is None:
                    start = base_index

                step = sub_i.step
                if step is None:
                    step = 1

                stop = sub_i.stop
                if stop is None:
                    stop = end_index

                if step == 1:
                    sub_i = Slice((
                            start - base_index,

                            # FIXME This is only correct for unit strides
                            stop - base_index + 1,

                            step
                            ))
                elif step == -1:
                    sub_i = Slice((
                            start - base_index,

                            # FIXME This is only correct for unit strides
                            stop - base_index - 1,

                            step
                            ))

                else:
                    # FIXME
                    raise NotImplementedError("Fortran slice processing for "
                            "non-unit strides")

            else:
                sub_i = sub_i - base_index

            new_subscript.append(sub_i)

        return expr.aggregate[self.rec(tuple(new_subscript))]

# }}}


# {{{ scope

class Scope:
    def __init__(self, subprogram_name, arg_names=None):
        if arg_names is None:
            arg_names = set()
        self.subprogram_name = subprogram_name

        # map first letter to type
        self.implicit_types = {}

        # map name to dim tuple
        self.dim_map = {}

        # map name to type
        self.type_map = {}

        # map name to data
        self.data_map = {}

        self.arg_names = arg_names

        self.index_sets = []

        # This dict has a key for every iname that is
        # currently active. These keys map to the loopy-side
        # expression for the iname, which may differ because
        # of non-zero lower iteration bounds or because of
        # duplicate inames need to be renamed for loopy.
        self.active_iname_aliases = {}

        self.active_loopy_inames = set()

        self.instructions = []
        self.temporary_variables = []

        self.used_names = set()

        self.previous_instruction_id = None

    def known_names(self):
        return (self.used_names
                | set(self.dim_map.keys())
                | set(self.type_map.keys()))

    def is_known(self, name):
        return (name in self.used_names
                or name in self.dim_map
                or name in self.type_map
                or name in self.arg_names)

    def all_inames(self):
        result = set()
        for iset in self.index_sets:
            result.update(iset.get_var_dict(dim_type.set))

        return frozenset(result)

    def use_name(self, name):
        self.used_names.add(name)

    def get_type(self, name, none_ok=False):
        try:
            return self.type_map[name]
        except KeyError:
            if self.implicit_types is None:
                if none_ok:
                    return None

                raise TranslationError(
                        "no type for '%s' found in 'implict none' routine"
                        % name)

            return self.implicit_types.get(name[0], np.dtype(np.int32))

    def get_loopy_shape(self, name):
        dims = self.dim_map.get(name, ())

        shape = []
        for i, dim in enumerate(dims):
            if len(dim) == 1:
                if isinstance(dim[0], Wildcard):
                    shape.append(None)
                else:
                    shape.append(dim[0])

            elif len(dim) == 2:
                if isinstance(dim[0], Wildcard):
                    shape.append(None)
                else:
                    shape.append(dim[1]-dim[0]+1)
            else:
                raise TranslationError("dimension axis %d "
                        "of '%s' not understood: %s"
                        % (i, name, dim))

        return tuple(shape)

    def process_expression_for_loopy(self, expr):
        from pymbolic.mapper.substitutor import make_subst_func
        from loopy.symbolic import SubstitutionMapper

        submap = SubstitutionMapper(
                make_subst_func(self.active_iname_aliases))

        expr = submap(expr)

        subshift = SubscriptIndexAdjuster(self)
        expr = subshift(expr)

        return expr

    def written_vars(self):
        return frozenset().union(*(insn.write_dependency_names()
                                   for insn in self.instructions))

    def read_vars(self):
        return (frozenset().union(*(insn.read_dependency_names()
                                   for insn in self.instructions))
                | frozenset().union(*(frozenset(bset.get_var_names(dim_type.param))
                                      for bset in self.index_sets)))

# }}}


# {{{ fortran division specializers

class FortranDivisionToFloorDiv(IdentityMapper):
    def map_fortran_division(self, expr, *args):
        from warnings import warn
        from loopy.diagnostic import LoopyWarning
        warn(
                "Integer division in Fortran do loop bound. "
                "Loopy currently forces this to integers and gets it wrong for "
                "negative arguments.", LoopyWarning)
        from pymbolic.primitives import FloorDiv
        return FloorDiv(
                self.rec(expr.numerator, *args),
                self.rec(expr.denominator, *args))


class FortranDivisionSpecializer(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, kernel, callables):
        super().__init__(rule_mapping_context)
        from loopy.type_inference import TypeReader
        self.infer_type = TypeReader(kernel, callables)
        self.kernel = kernel

    def map_fortran_division(self, expr, *args):
        # We remove all these before type inference ever sees them.
        from loopy.type_inference import TypeInferenceFailure

        try:
            num_dtype = self.infer_type(expr.numerator).numpy_dtype
            den_dtype = self.infer_type(expr.denominator).numpy_dtype
        except TypeInferenceFailure:
            return super().map_fortran_division(expr, *args)

        from pymbolic.primitives import Quotient, FloorDiv
        if num_dtype.kind in "iub" and den_dtype.kind in "iub":
            warn_with_kernel(self.kernel,
                    "fortran_int_div",
                    "Integer division in Fortran code. Loopy currently gets this "
                    "wrong for negative arguments.")
            return FloorDiv(
                    self.rec(expr.numerator, *args),
                    self.rec(expr.denominator, *args))

        else:
            return Quotient(
                    self.rec(expr.numerator, *args),
                    self.rec(expr.denominator, *args))


def _specialize_fortran_division_for_kernel(knl, callables):
    rmc = SubstitutionRuleMappingContext(
            knl.substitutions, knl.get_var_name_generator())
    return FortranDivisionSpecializer(rmc, knl, callables).map_kernel(knl)


def specialize_fortran_division(t_unit):
    from loopy.translation_unit import TranslationUnit, resolve_callables
    from loopy.kernel.function_interface import CallableKernel
    from loopy.type_inference import infer_unknown_types
    assert isinstance(t_unit, TranslationUnit)

    t_unit = resolve_callables(t_unit)
    t_unit = infer_unknown_types(t_unit)
    new_callables = {}

    for name, clbl in t_unit.callables_table.items():
        if isinstance(clbl, CallableKernel):
            knl = clbl.subkernel
            clbl = clbl.copy(subkernel=_specialize_fortran_division_for_kernel(
                    knl, t_unit.callables_table))

        new_callables[name] = clbl

    return t_unit.copy(callables_table=Map(new_callables))

# }}}


# {{{ translator

class F2LoopyTranslator(FTreeWalkerBase):
    def __init__(self, filename, target=None):
        FTreeWalkerBase.__init__(self, filename)

        self.target = target

        self.scope_stack = []

        self.insn_id_counter = 0
        self.condition_id_counter = 0

        self.kernels = []

        self.instruction_tags = []
        self.conditions = []
        self.conditions_data = []

        self.index_dtype = None

        self.block_nest = []

    def add_instruction(self, insn):
        scope = self.scope_stack[-1]

        scope.previous_instruction_id = insn.id
        scope.instructions.append(insn)

    def add_expression_instruction(self, lhs, rhs):
        scope = self.scope_stack[-1]

        new_id = self.get_insn_id()

        from loopy.kernel.data import Assignment
        insn = Assignment(
                lhs, rhs,
                within_inames=frozenset(
                    scope.active_loopy_inames),
                id=new_id,
                predicates=frozenset(self.conditions),
                tags=tuple(self.instruction_tags))

        self.add_instruction(insn)

    def get_insn_id(self):
        new_id = intern("insn%d" % self.insn_id_counter)
        self.insn_id_counter += 1

        return new_id

    # {{{ map_XXX functions

    def map_BeginSource(self, node):
        scope = Scope(None)
        self.scope_stack.append(scope)

        for c in node.content:
            self.rec(c)

    def map_Subroutine(self, node):
        assert not node.prefix
        assert not hasattr(node, "suffix")

        scope = Scope(node.name, list(node.args))
        self.scope_stack.append(scope)

        self.block_nest.append("sub")
        for c in node.content:
            self.rec(c)

        self.scope_stack.pop()

        self.kernels.append(scope)

    def map_EndSubroutine(self, node):
        if not self.block_nest:
            raise TranslationError("no subroutine started at this point")
        if self.block_nest.pop() != "sub":
            raise TranslationError("mismatched end subroutine")

        return []

    def map_Implicit(self, node):
        scope = self.scope_stack[-1]

        if not node.items:
            assert not scope.implicit_types
            scope.implicit_types = None

        for stmt, specs in node.items:
            if scope.implict_types is None:
                raise TranslationError("implicit decl not allowed after "
                        "'implicit none'")
            tp = self.dtype_from_stmt(stmt)
            for start, end in specs:
                for char_code in range(ord(start), ord(end)+1):
                    scope.implicit_types[chr(char_code)] = tp

        return []

    # {{{ types, declarations

    def map_Equivalence(self, node):
        raise NotImplementedError("equivalence")

    TYPE_MAP = {
            ("real", ""): np.float32,
            ("real", "4"): np.float32,
            ("real", "8"): np.float64,
            ("doubleprecision", ""): np.float64,

            ("complex", "8"): np.complex64,
            ("complex", "16"): np.complex128,

            ("integer", ""): np.int32,
            ("integer", "4"): np.int32,
            ("integer", "8"): np.int64,
            }
    if hasattr(np, "float128"):
        TYPE_MAP[("real", "16")] = np.float128  # pylint:disable=no-member
    if hasattr(np, "complex256"):
        TYPE_MAP[("complex", "32")] = np.complex256  # pylint:disable=no-member

    def dtype_from_stmt(self, stmt):
        length, kind = stmt.selector

        if kind and not length:
            length = kind
        elif length and not kind:
            pass
        elif not length and not kind:
            pass
        else:
            raise RuntimeError("both length and kind specified")

        return np.dtype(self.TYPE_MAP[(type(stmt).__name__.lower(), length)])

    def map_type_decl(self, node):
        scope = self.scope_stack[-1]

        tp = self.dtype_from_stmt(node)

        for name, shape, initializer in self.parse_dimension_specs(
                node, node.entity_decls):
            if shape is not None:
                assert name not in scope.dim_map
                scope.dim_map[name] = shape
                scope.use_name(name)

            assert name not in scope.type_map
            scope.type_map[name] = tp

            assert name not in scope.data_map
            scope.data_map[name] = initializer

        return []

    map_Logical = map_type_decl  # noqa: N815
    map_Integer = map_type_decl  # noqa: N815
    map_Real = map_type_decl  # noqa: N815
    map_Complex = map_type_decl  # noqa: N815
    map_DoublePrecision = map_type_decl  # noqa: N815

    def map_Dimension(self, node):
        scope = self.scope_stack[-1]

        for name, shape, initializer in self.parse_dimension_specs(node, node.items):
            if initializer is not None:
                raise LoopyError("initializer in dimension statement")

            if shape is not None:
                assert name not in scope.dim_map
                scope.dim_map[name] = shape
                scope.use_name(name)

        return []

    def map_External(self, node):
        raise NotImplementedError("external")

    # }}}

    def map_Data(self, node):
        scope = self.scope_stack[-1]

        for name, data in node.stmts:
            name, = name
            assert name not in scope.data
            scope.data[name] = [
                    scope.process_expression_for_loopy(
                        self.parse_expr(node, i)) for i in data]

        return []

    def map_Parameter(self, node):
        raise NotImplementedError("parameter")

    # {{{ I/O

    def map_Open(self, node):
        raise NotImplementedError

    def map_Format(self, node):
        warn("'format' unsupported", TranslatorWarning)

    def map_Write(self, node):
        warn("'write' unsupported", TranslatorWarning)

    def map_Print(self, node):
        warn("'print' unsupported", TranslatorWarning)

    def map_Read1(self, node):
        warn("'read' unsupported", TranslatorWarning)

    # }}}

    def map_Assignment(self, node):
        scope = self.scope_stack[-1]

        lhs = scope.process_expression_for_loopy(
                self.parse_expr(node, node.variable))
        from pymbolic.primitives import Subscript, Call
        if isinstance(lhs, Call):
            raise TranslationError("function call (to '%s') on left hand side of"
                    "assignment--check for misspelled variable name" % lhs)
        elif isinstance(lhs, Subscript):
            lhs_name = lhs.aggregate.name
        else:
            lhs_name = lhs.name

        scope.use_name(lhs_name)

        rhs = scope.process_expression_for_loopy(self.parse_expr(node, node.expr))

        self.add_expression_instruction(lhs, rhs)

    def map_Allocate(self, node):
        raise NotImplementedError("allocate")

    def map_Deallocate(self, node):
        raise NotImplementedError("deallocate")

    def map_Save(self, node):
        raise NotImplementedError("save")

    def map_Line(self, node):
        pass

    def map_Program(self, node):
        raise NotImplementedError

    def map_Entry(self, node):
        raise NotImplementedError("entry")

    # {{{ control flow

    def map_Goto(self, node):
        raise NotImplementedError("goto")

    def map_Call(self, node):
        from loopy.kernel.instruction import _get_assignee_var_name
        scope = self.scope_stack[-1]

        new_id = self.get_insn_id()

        # {{{ comply with loopy's kernel call requirements

        callee, = (knl for knl in self.kernels
                   if knl.subprogram_name == node.designator)
        call_params = [scope.process_expression_for_loopy(self.parse_expr(node,
                                                                          item))
                       for item in node.items]
        callee_read_vars = callee.read_vars()
        callee_written_vars = callee.written_vars()

        lpy_params = []
        lpy_assignees = []
        for param in call_params:
            name = _get_assignee_var_name(param)
            if name in callee_read_vars:
                lpy_params.append(param)
            if name in callee_written_vars:
                lpy_assignees.append(param)
            if name not in (callee_read_vars | callee_written_vars):
                lpy_params.append(param)

        # }}}

        from pymbolic import var

        from loopy.kernel.data import CallInstruction
        insn = CallInstruction(
                tuple(lpy_assignees),
                var(node.designator)(*lpy_params),
                within_inames=frozenset(
                    scope.active_loopy_inames),
                id=new_id,
                predicates=frozenset(self.conditions),
                tags=tuple(self.instruction_tags))

        self.add_instruction(insn)

    def map_Return(self, node):
        raise NotImplementedError("return")

    def map_ArithmeticIf(self, node):
        raise NotImplementedError("arithmetic-if")

    def realize_conditional(self, node, context_cond=None):
        scope = self.scope_stack[-1]

        cond_name = intern("loopy_cond%d" % self.condition_id_counter)
        self.condition_id_counter += 1
        assert cond_name not in scope.type_map

        scope.type_map[cond_name] = np.int32

        from pymbolic import var
        cond_var = var(cond_name)

        self.add_expression_instruction(
                cond_var,
                scope.process_expression_for_loopy(
                    self.parse_expr(node, node.expr)))

        cond_expr = cond_var
        if context_cond is not None:
            from pymbolic.primitives import LogicalAnd
            cond_expr = LogicalAnd((cond_var, context_cond))

            self.conditions_data.append((context_cond, cond_var))
        else:
            self.conditions_data.append((None, cond_var))

        self.conditions.append(cond_expr)

    def map_If(self, node):
        self.realize_conditional(node, None)

        for c in node.content:
            self.rec(c)

        self.conditions_data.pop()
        self.conditions.pop()

    def map_IfThen(self, node):
        self.block_nest.append("if")
        self.realize_conditional(node, None)

        for c in node.content:
            self.rec(c)

    def construct_else_condition(self):
        context_cond, prev_cond = self.conditions_data.pop()
        if prev_cond is None:
            raise RuntimeError("else if may not follow else")

        self.conditions.pop()

        from pymbolic.primitives import LogicalNot, LogicalAnd
        else_expr = LogicalNot(prev_cond)
        if context_cond is not None:
            else_expr = LogicalAnd((else_expr, context_cond))

        return else_expr

    def map_Else(self, node):
        else_cond = self.construct_else_condition()
        self.conditions.append(else_cond)
        self.conditions_data.append((else_cond, None))

    def map_ElseIf(self, node):
        self.realize_conditional(node, self.construct_else_condition())

    def map_EndIfThen(self, node):
        if not self.block_nest:
            raise TranslationError("no if block started at end if")
        if self.block_nest.pop() != "if":
            raise TranslationError("mismatched end if")

        self.conditions_data.pop()
        self.conditions.pop()

    def map_Do(self, node):
        scope = self.scope_stack[-1]

        if not node.loopcontrol:
            raise NotImplementedError("unbounded do loop")

        loop_var, loop_bounds = node.loopcontrol.split("=")
        loop_var = loop_var.strip()

        iname_dtype = scope.get_type(loop_var)
        if self.index_dtype is None:
            self.index_dtype = iname_dtype
        else:
            if self.index_dtype != iname_dtype:
                raise LoopyError("type of '%s' (%s) does not agree with prior "
                        "index type (%s)"
                        % (loop_var, iname_dtype, self.index_dtype))

        scope.use_name(loop_var)
        loop_bounds = scope.process_expression_for_loopy(
                self.parse_expr(
                    node,
                    loop_bounds, min_precedence=self.expr_parser._PREC_FUNC_ARGS))

        if len(loop_bounds) == 2:
            start, stop = loop_bounds
            step = 1
        elif len(loop_bounds) == 3:
            start, stop, step = loop_bounds
        else:
            raise RuntimeError("loop bounds not understood: %s"
                    % node.loopcontrol)

        if step != 1:
            raise NotImplementedError(
                    "do loops with non-unit stride")

        if not isinstance(step, int):
            raise TranslationError(
                    "non-constant steps not supported: %s" % step)

        from loopy.symbolic import get_dependencies
        loop_bound_deps = (
                get_dependencies(start)
                | get_dependencies(stop)
                | get_dependencies(step))

        # {{{ find a usable loopy-side loop name

        loopy_loop_var = loop_var
        loop_var_suffix = None
        while True:
            already_used = False
            for iset in scope.index_sets:
                if loopy_loop_var in iset.get_var_dict(dim_type.set):
                    already_used = True
                    break

            if not already_used:
                break

            if loop_var_suffix is None:
                loop_var_suffix = 0

            loop_var_suffix += 1
            loopy_loop_var = loop_var + "_%d" % loop_var_suffix

        loopy_loop_var = intern(loopy_loop_var)

        # }}}

        space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT,
                set=[loopy_loop_var], params=list(loop_bound_deps))

        from loopy.isl_helpers import iname_rel_aff
        from loopy.symbolic import aff_from_expr
        index_set = (
                isl.BasicSet.universe(space)
                .add_constraint(
                    isl.Constraint.inequality_from_aff(
                        iname_rel_aff(space,
                            loopy_loop_var, ">=",
                            aff_from_expr(space, 0))))
                .add_constraint(
                    isl.Constraint.inequality_from_aff(
                        iname_rel_aff(space,
                            loopy_loop_var, "<=",
                            aff_from_expr(space, FortranDivisionToFloorDiv()(
                                stop-start))))))

        from pymbolic import var
        scope.active_iname_aliases[loop_var] = \
                var(loopy_loop_var) + start
        scope.active_loopy_inames.add(loopy_loop_var)

        scope.index_sets.append(index_set)

        self.block_nest.append("do")

        for c in node.content:
            self.rec(c)

        del scope.active_iname_aliases[loop_var]
        scope.active_loopy_inames.remove(loopy_loop_var)

    def map_EndDo(self, node):
        if not self.block_nest:
            raise TranslationError("no do loop started at end do")
        if self.block_nest.pop() != "do":
            raise TranslationError("mismatched end do")

    def map_Continue(self, node):
        raise NotImplementedError("continue")

    def map_Stop(self, node):
        raise NotImplementedError("stop")

    faulty_loopy_pragma = re.compile(r"\s*\$\s*loopy\s*")

    begin_tag_re = re.compile(r"\$loopy begin tagged:\s*(.*?)\s*$")
    end_tag_re = re.compile(r"\$loopy end tagged:\s*(.*?)\s*$")

    def map_Comment(self, node):
        stripped_comment_line = node.content.strip()

        begin_tag_match = self.begin_tag_re.match(stripped_comment_line)
        end_tag_match = self.end_tag_re.match(stripped_comment_line)
        faulty_loopy_pragma_match = self.faulty_loopy_pragma.match(
                stripped_comment_line)

        if begin_tag_match:
            tag = LegacyStringInstructionTag(begin_tag_match.group(1))
            if tag in self.instruction_tags:
                raise TranslationError(f"nested begin tag for tag '{tag.value}'")
            self.instruction_tags.append(tag)

        elif end_tag_match:
            tag = LegacyStringInstructionTag(end_tag_match.group(1))
            if tag not in self.instruction_tags:
                raise TranslationError(
                        f"end tag without begin tag for tag '{tag.value}'")
            self.instruction_tags.remove(tag)

        elif faulty_loopy_pragma_match is not None:
            from warnings import warn
            warn("The comment line '%s' was not recognized as a loopy directive"
                    % stripped_comment_line)

    # }}}

    # }}}

    def make_kernels(self, seq_dependencies):
        result = []

        for sub in self.kernels:
            # {{{ figure out arguments

            kernel_data = []
            for arg_name in sub.arg_names:
                dims = sub.dim_map.get(arg_name)

                if sub.data_map.get(arg_name) is not None:
                    raise NotImplementedError(
                            "initializer for argument %s" % arg_name)

                if dims is not None:
                    # default order is set to "F" in kernel creation below
                    kernel_data.append(
                            lp.GlobalArg(
                                arg_name,
                                dtype=sub.get_type(arg_name),
                                shape=sub.get_loopy_shape(arg_name),
                                ))
                else:
                    kernel_data.append(
                            lp.ValueArg(arg_name,
                                dtype=sub.get_type(arg_name)))

            # }}}

            # {{{ figure out temporary variables

            for var_name in (
                    sub.known_names()
                    - set(sub.arg_names)
                    - sub.all_inames()):
                dtype = sub.get_type(var_name, none_ok=True)
                if sub.implicit_types is None and dtype is None:
                    continue

                kwargs = {}
                if sub.data_map.get(var_name) is not None:
                    kwargs["read_only"] = True
                    kwargs["address_space"] = lp.AddressSpace.PRIVATE
                    kwargs["initializer"] = np.array(
                            sub.data_map[var_name], dtype=dtype)

                kernel_data.append(
                        lp.TemporaryVariable(
                            var_name, dtype=dtype,
                            shape=sub.get_loopy_shape(var_name),
                            **kwargs))

            # }}}

            knl = lp.make_function(
                    sub.index_sets,
                    sub.instructions,
                    kernel_data,
                    name=sub.subprogram_name,
                    default_order="F",
                    index_dtype=self.index_dtype,
                    target=self.target,
                    seq_dependencies=seq_dependencies,
                    lang_version=(2018, 2),
                    )

            from loopy.loop import merge_loop_domains
            knl = merge_loop_domains(knl)

            knl = lp.fold_constants(knl)

            result.append(knl)

        return result

# }}}

# vim: foldmethod=marker

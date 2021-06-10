__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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

from pymbolic.mapper.differentiator import DifferentiationMapper

import pymbolic.primitives as p
var = p.Variable

import loopy as lp
from loopy.symbolic import RuleAwareIdentityMapper, SubstitutionRuleMappingContext
from loopy.isl_helpers import make_slab
from loopy.diagnostic import LoopyError
from loopy.kernel import LoopKernel


# {{{ diff mapper

def func_map(i, func, args, allowed_nonsmoothness):
    if func.name == "exp":
        return var("exp")(*args)
    elif func.name == "log":
        return 1/args[0]

    elif func.name == "sin":
        return var("cos")(*args)
    elif func.name == "cos":
        return -var("sin")(*args)
    elif func.name == "tan":
        return 1+var("tan")(*args)**2

    elif func.name == "sinh":
        return var("cosh")(*args)
    elif func.name == "cosh":
        return var("sinh")(*args)
    elif func.name == "tanh":
        return (1 - var("tanh")(*args))**2

    else:
        raise NotImplementedError("derivative of '%s'" % func.name)


class LoopyDiffMapper(DifferentiationMapper, RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, diff_context, diff_inames,
            allowed_nonsmoothness=None):
        RuleAwareIdentityMapper.__init__(self, rule_mapping_context)
        DifferentiationMapper.__init__(
                self,

                # This is actually ignored because we
                # override map_variable below.
                variable=None,

                allowed_nonsmoothness=None)
        self.diff_context = diff_context
        self.diff_inames = diff_inames
        self.diff_iname_exprs = tuple(var(diname) for diname in diff_inames)
        self.function_map = func_map

    def rec_undiff(self, expr, *args):
        dc = self.diff_context

        from loopy.symbolic import get_dependencies
        deps = get_dependencies(expr)
        for dep in deps:
            assert isinstance(dep, str)
            if (dep in dc.kernel.arg_dict
                    or dep in dc.kernel.temporary_variables):
                dc.import_output_var(dep)

        return expr

    def map_variable(self, expr, *args):
        dc = self.diff_context

        if expr.name == dc.by_name:
            assert len(self.diff_inames) == 0
            return 1

        elif (expr.name in dc.kernel.all_inames()
                or expr.name in dc.kernel.all_params()):
            return expr

        else:
            dvar_dby = dc.get_diff_var(expr.name)
            if dvar_dby is None:
                return 0

            if self.diff_inames:
                return var(dvar_dby)[self.diff_iname_exprs]
            else:
                return var(dvar_dby)

    map_tagged_variable = map_variable

    def map_subscript(self, expr, *args):
        dc = self.diff_context

        if expr.aggregate.name == dc.by_name:
            index = expr.index
            if not isinstance(expr.index, tuple):
                index = (expr.index,)

            assert len(self.diff_inames) == len(index)

            conds = [
                p.Comparison(var(ti), "==", ei)
                for ti, ei in zip(self.diff_inames, index)
                ]

            if len(conds) == 1:
                and_conds, = conds
            elif len(conds) > 1:
                and_conds = p.LogicalAnd(tuple(conds))
            else:
                raise AssertionError()

            return p.If(and_conds, 1, 0)

        else:
            dvar_dby = dc.get_diff_var(expr.aggregate.name)
            if dvar_dby is None:
                return 0

            idx = expr.index
            if not isinstance(idx, tuple):
                idx = (idx,)

            return type(expr)(
                    var(dvar_dby),
                    expr.index + self.diff_inames)

    def map_call(self, expr, *args):
        dc = self.diff_context

        if expr.function.name in dc.kernel.substitutions:
            # FIXME: Deal with subsitution rules
            # Need to use chain rule here, too.
            raise NotImplementedError("substitution rules in differentiation")
        else:
            return DifferentiationMapper.map_call(self, expr, *args)

# }}}


# {{{ diff context

class DifferentiationContext:
    def __init__(self, kernel, var_name_gen, by_name, diff_iname_prefix,
            additional_shape):
        self.kernel = kernel
        self.by_name = by_name
        self.diff_iname_prefix = diff_iname_prefix
        self.additional_shape = additional_shape

        self.imported_outputs = set()
        self.output_to_diff_output = {}

        self.generate_instruction_id = self.kernel.get_instruction_id_generator()

        self.new_args = []
        self.new_temporary_variables = {}
        self.new_instructions = []
        self.imported_instructions = set()
        self.new_domains = []

        self.rule_mapping_context = SubstitutionRuleMappingContext(
                kernel.substitutions, var_name_gen)

    def get_new_kernel(self):
        knl = self.kernel

        new_args = knl.args + self.new_args
        new_temp_vars = knl.temporary_variables.copy()
        new_temp_vars.update(self.new_temporary_variables)

        knl = knl.copy(
                args=new_args,
                temporary_variables=new_temp_vars,
                instructions=self.new_instructions,
                domains=knl.domains + self.new_domains)

        del new_args
        del new_temp_vars

        knl = self.rule_mapping_context.finish_kernel(knl)

        return knl

    # {{{ kernel gen entrypoints

    def add_diff_inames(self):
        diff_inames = tuple(
            self.rule_mapping_context.make_unique_var_name(
                self.diff_iname_prefix+str(i))
            for i in range(len(self.additional_shape)))

        diff_parameters = set()
        from loopy.symbolic import get_dependencies
        for s in self.additional_shape:
            diff_parameters.update(get_dependencies(s))

        diff_domain = isl.BasicSet(
                "[%s] -> {[%s]}"
                % (", ".join(diff_parameters), ", ".join(diff_inames)))

        for i, diff_iname in enumerate(diff_inames):
            diff_domain = diff_domain & make_slab(
                diff_domain.space, diff_iname, 0, self.additional_shape[i])

        self.new_domains.append(diff_domain)

        return diff_inames

    # }}}

    def import_instruction_and_deps(self, insn_id):
        if insn_id in self.imported_instructions:
            return

        insn = self.kernel.id_to_insn[insn_id]
        self.new_instructions.append(insn)
        self.imported_instructions.add(insn_id)

        id_map = RuleAwareIdentityMapper(self.rule_mapping_context)

        if isinstance(insn, lp.Assignment):
            id_map(insn.expression, self.kernel, insn)
        else:
            raise RuntimeError("do not know how to deal with "
                    "instruction of type %s" % type(insn))

        for dep in insn.depends_on:
            self.import_instruction_and_deps(dep)

    def import_output_var(self, var_name):
        writers = self.kernel.writer_map().get(var_name, [])

        if len(writers) > 1:
            raise LoopyError("%s is written in more than one place"
                    % var_name)

        if not writers:
            return

        insn_id, = writers
        self.import_instruction_and_deps(insn_id)

    def get_diff_var(self, var_name):
        """
        :return: a string containing the name of a new variable
            holding the derivative of *var_name* by the desired
            *diff_context.by_name*, or *None* if no dependency exists.
        """
        new_var_name = self.rule_mapping_context.make_unique_var_name(
                var_name + "_d" + self.by_name)

        writers = self.kernel.writer_map().get(var_name, [])

        if not writers:
            # FIXME: There should be hooks to supply earlier dvar_dby
            # This would be the spot to think about them.
            return None

        if len(writers) > 1:
            raise LoopyError("%s is written in more than one place"
                    % var_name)

        orig_writer_id, = writers
        orig_writer_insn = self.kernel.id_to_insn[orig_writer_id]

        diff_inames = self.add_diff_inames()
        diff_iname_exprs = tuple(var(diname) for diname in diff_inames)

        # {{{ write code

        diff_mapper = LoopyDiffMapper(self.rule_mapping_context, self,
                diff_inames)

        diff_expr = diff_mapper(orig_writer_insn.expression,
                self.kernel, orig_writer_insn)

        if not diff_expr:
            return None

        assert isinstance(orig_writer_insn, lp.Assignment)
        if isinstance(orig_writer_insn.assignee, p.Subscript):
            lhs_ind = orig_writer_insn.assignee.index_tuple
        elif isinstance(orig_writer_insn.assignee, p.Variable):
            lhs_ind = ()
        else:
            raise LoopyError(
                    "Unrecognized LHS type in differentiation: %s"
                    % type(orig_writer_insn.assignee).__name__)

        new_insn_id = self.generate_instruction_id()
        insn = lp.Assignment(
                id=new_insn_id,
                assignee=var(new_var_name)[
                    lhs_ind + diff_iname_exprs],
                expression=diff_expr,
                within_inames=(
                    orig_writer_insn.within_inames | frozenset(diff_inames)))

        self.new_instructions.append(insn)

        # }}}

        # {{{ manage variable declaration

        if var_name in self.kernel.arg_dict:
            arg = self.kernel.arg_dict[var_name]
            orig_shape = arg.shape

        elif var_name in self.kernel.temporary_variables:
            tv = self.kernel.temporary_variables[var_name]
            orig_shape = tv.shape

        else:
            raise ValueError("%s: variable not found" % var_name)

        shape = orig_shape + self.additional_shape
        dim_tags = ("c",) * len(shape)

        if var_name in self.kernel.arg_dict:
            self.new_args.append(
                lp.GlobalArg(
                    new_var_name,
                    arg.dtype,
                    shape=shape,
                    dim_tags=dim_tags,
                    is_input=arg.is_input,
                    is_output=arg.is_output
                ))

        elif var_name in self.kernel.temporary_variables:
            self.new_temporary_variables[new_var_name] = lp.TemporaryVariable(
                    new_var_name,
                    tv.dtype,
                    shape=shape,
                    dim_tags=dim_tags)

        # }}}

        return new_var_name

# }}}


# {{{ entrypoint

def diff_kernel(kernel, diff_outputs, by, diff_iname_prefix="diff_i",
        batch_axes_in_by=frozenset(), copy_outputs=frozenset()):
    """

    :arg batch_axes_in_by: a :class:`set` of axis indices in the variable named *by*
        that are not part of the differentiation.
    :return: a string containing the name of a new variable
        holding the derivative of *var_name* by the desired
        *diff_context.by_name*, or *None* if no dependency exists.
    """

    assert isinstance(kernel, LoopKernel)

    from loopy.kernel.creation import apply_single_writer_depencency_heuristic
    kernel = apply_single_writer_depencency_heuristic(kernel, warn_if_used=True)

    if isinstance(diff_outputs, str):
        diff_outputs = [
                dout.strip() for dout in diff_outputs.split(",")
                if dout.strip()]

    by_arg = kernel.arg_dict[by]
    additional_shape = by_arg.shape

    var_name_gen = kernel.get_var_name_generator()

    # {{{ differentiate instructions

    diff_context = DifferentiationContext(
            kernel, var_name_gen, by, diff_iname_prefix=diff_iname_prefix,
            additional_shape=additional_shape)

    result = {}
    for dout in diff_outputs:
        result = diff_context.get_diff_var(dout)

    for cout in copy_outputs:
        diff_context.import_output_var(cout)

    # }}}

    return diff_context.get_new_kernel(), result

# }}}


# vim: foldmethod=marker

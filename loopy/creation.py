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
from loopy.symbolic import IdentityMapper


def tag_reduction_inames_as_sequential(knl):
    result = set()

    def map_reduction(red_expr, rec):
        rec(red_expr.expr)
        result.update(red_expr.inames)

    from loopy.symbolic import ReductionCallbackMapper
    for insn in knl.instructions:
        ReductionCallbackMapper(map_reduction)(insn.expression)

    from loopy.kernel import ParallelTag, ForceSequentialTag

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
            raise RuntimeError("there must be exactly one write to data-dependent "
                    "domain parameter '%s' (found %d)" % (tvpar, len(par_writers)))


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
            from loopy import infer_type
            dtype = infer_type
        else:
            dtype=np.dtype(dtype)

        from loopy.kernel import TemporaryVariable
        new_temp_vars[new_var_name] = TemporaryVariable(
                name=new_var_name,
                dtype=dtype,
                is_local=None,
                shape=())

        from pymbolic.primitives import Variable
        from loopy.kernel import Instruction
        insn = Instruction(
                id=knl.make_unique_instruction_id(extra_used_ids=newly_created_insn_ids),
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

    for insn in knl.instructions:
        from loopy.kernel import TemporaryVariable

        if insn.temp_var_type is not None:
            assignee_name = insn.get_assignee_var_name()

            assignee_indices = []
            from pymbolic.primitives import Variable
            for index_expr in insn.get_assignee_indices():
                if (not isinstance(index_expr, Variable)
                        or not index_expr.name in knl.all_inames()):
                    raise RuntimeError(
                            "only plain inames are allowed in "
                            "the lvalue index when declaring the "
                            "variable '%s' in an instruction"
                            % assignee_name)

                assignee_indices.append(index_expr.name)

            base_indices, shape = \
                    knl.find_var_base_indices_and_shape_from_inames(
                            assignee_indices, knl.cache_manager)

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
                raise RuntimeError("Reduction iname duplication with '@' is no "
                        "longer supported. Use loopy.duplicate_inames instead.")

    # }}}


    from loopy.symbolic import ReductionCallbackMapper
    rcm = ReductionCallbackMapper(check_reduction_inames)
    for insn in kernel.instructions:
        rcm(insn.expression)

    for sub_name, sub_rule in kernel.substitutions.iteritems():
        rcm(sub_rule.expression)

# }}}

# {{{ kernel creation top-level

def make_kernel(*args, **kwargs):
    """Second pass of kernel creation. Think about requests for iname duplication
    and temporary variable creation.
    """

    from loopy.kernel import LoopKernel
    knl = LoopKernel(*args, **kwargs)

    from loopy import tag_inames
    knl = tag_inames(
            knl.copy(iname_to_tag_requests=None),
            knl.iname_to_tag_requests).copy(
                    iname_to_tag_requests=[])

    check_for_nonexistent_iname_deps(knl)
    check_for_reduction_inames_duplication_requests(knl)


    knl = tag_reduction_inames_as_sequential(knl)
    knl = create_temporaries(knl)
    knl = expand_cses(knl)

    # -------------------------------------------------------------------------
    # Ordering dependency:
    # -------------------------------------------------------------------------
    # Must create temporary before checking for writes to temporary variables
    # that are domain parameters.
    # -------------------------------------------------------------------------

    check_for_multiple_writes_to_loop_bounds(knl)
    check_for_duplicate_names(knl)
    check_written_variable_names(knl)

    return knl

# }}}

# vim: fdm=marker

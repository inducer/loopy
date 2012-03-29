from __future__ import division
import numpy as np
from loopy.symbolic import IdentityMapper


# {{{ sanity checking

def check_kernel(knl):
    for insn in knl.instructions:
        if not set(insn.forced_iname_deps) <= knl.all_inames():
            raise ValueError("In instruction '%s': "
                    "cannot force dependency on inames '%s'--"
                    "they don't exist" % (
                        insn.id,
                        ",".join(
                            set(insn.forced_iname_deps)-knl.all_inames())))

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

            var_name = self.add_assignment(expr.prefix, child, dtype)
            from pymbolic.primitives import Variable
            var = Variable(var_name)
            self.expr_to_var[expr.child] = var
            return var

def expand_cses(knl):
    def add_assignment(base_name, expr, dtype):
        kwargs = dict(extra_used_vars=newly_created_vars)
        if base_name is not None:
            kwargs["based_on"] = base_name
        new_var_name = knl.make_unique_var_name(**kwargs)
        newly_created_vars.add(new_var_name)

        if dtype is None:
            dtype = tim(expr)

        from loopy.kernel import TemporaryVariable
        new_temp_vars[new_var_name] = TemporaryVariable(
                name=new_var_name,
                dtype=np.dtype(dtype),
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

    newly_created_vars = set()
    newly_created_insn_ids = set()
    new_temp_vars = knl.temporary_variables.copy()

    from loopy.codegen.expression import TypeInferenceMapper
    tim = TypeInferenceMapper(knl, new_temp_vars)

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
        from loopy.kernel import (
                find_var_base_indices_and_shape_from_inames,
                TemporaryVariable)

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
                    find_var_base_indices_and_shape_from_inames(
                            knl.domain, assignee_indices, knl.cache_manager)

            new_temp_vars[assignee_name] = TemporaryVariable(
                    name=assignee_name,
                    dtype=np.dtype(insn.temp_var_type),
                    is_local=None,
                    base_indices=base_indices,
                    shape=shape)

            insn = insn.copy(temp_var_type=None)

        new_insns.append(insn)

    return knl.copy(
            instructions=new_insns,
            temporary_variables=new_temp_vars)

# }}}

# {{{ duplicate inames

def duplicate_inames(knl):
    from loopy.kernel import AutoFitLocalIndexTag

    new_insns = []
    new_domain = knl.domain
    new_iname_to_tag = knl.iname_to_tag.copy()

    newly_created_vars = set()

    for insn in knl.instructions:
        if insn.duplicate_inames_and_tags:
            insn_dup_iname_to_tag = dict(insn.duplicate_inames_and_tags)

            if not set(insn_dup_iname_to_tag.keys()) <= knl.all_inames():
                raise ValueError("In instruction '%s': "
                        "cannot duplicate inames '%s'--"
                        "they don't exist" % (
                            insn.id,
                            ",".join(
                                set(insn_dup_iname_to_tag.keys())-knl.all_inames())))

            # {{{ duplicate non-reduction inames

            reduction_inames = insn.reduction_inames()

            inames_to_duplicate = [iname
                    for iname, tag in insn.duplicate_inames_and_tags
                    if iname not in reduction_inames]

            new_inames = [
                    knl.make_unique_var_name(
                        based_on=iname+"_"+insn.id,
                        extra_used_vars=newly_created_vars)
                    for iname in inames_to_duplicate]

            for old_iname, new_iname in zip(inames_to_duplicate, new_inames):
                new_tag = insn_dup_iname_to_tag[old_iname]
                if new_tag is None:
                    new_tag = AutoFitLocalIndexTag()
                new_iname_to_tag[new_iname] = new_tag

            newly_created_vars.update(new_inames)

            from loopy.isl_helpers import duplicate_axes
            new_domain = duplicate_axes(new_domain, inames_to_duplicate, new_inames)

            from loopy.symbolic import SubstitutionMapper
            from pymbolic.mapper.substitutor import make_subst_func
            from pymbolic import var
            old_to_new = dict(
                    (old_iname, var(new_iname))
                    for old_iname, new_iname in zip(inames_to_duplicate, new_inames))
            subst_map = SubstitutionMapper(make_subst_func(old_to_new))
            new_expression = subst_map(insn.expression)

            # }}}

            if len(inames_to_duplicate) < len(insn.duplicate_inames_and_tags):
                raise RuntimeError("cannot use [|...] syntax to rename reduction "
                        "inames")

            insn = insn.copy(
                    assignee=subst_map(insn.assignee),
                    expression=new_expression,
                    forced_iname_deps=set(
                        old_to_new.get(iname, iname) for iname in insn.forced_iname_deps),
                    duplicate_inames_and_tags=[])

        new_insns.append(insn)

    return knl.copy(
            instructions=new_insns,
            domain=new_domain,
            iname_to_tag=new_iname_to_tag)
# }}}

# {{{ kernel creation top-level

def make_kernel(*args, **kwargs):
    """Second pass of kernel creation. Think about requests for iname duplication
    and temporary variable declaration received as part of string instructions.
    """

    from loopy.kernel import LoopKernel
    knl = LoopKernel(*args, **kwargs)

    from loopy import tag_dimensions
    knl = tag_dimensions(
            knl.copy(iname_to_tag_requests=None),
            knl.iname_to_tag_requests).copy(
                    iname_to_tag_requests=[])

    check_kernel(knl)

    knl = create_temporaries(knl)
    knl = expand_cses(knl)
    knl = duplicate_inames(knl)

    return knl

# }}}

# vim: fdm=marker

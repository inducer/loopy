from __future__ import division
import numpy as np
from loopy.symbolic import IdentityMapper


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
        kwargs = dict(extra_used_vars=newly_created_vars)
        if base_name is not None:
            kwargs["based_on"] = base_name
        new_var_name = knl.make_unique_var_name(**kwargs)
        newly_created_vars.add(new_var_name)

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

    newly_created_vars = set()
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

# {{{ reduction iname duplication

def duplicate_reduction_inames(kernel):

    # {{{ helper function

    newly_created_vars = set()

    def duplicate_reduction_inames(reduction_expr, rec):
        child = rec(reduction_expr.expr)
        new_red_inames = []
        did_something = False

        for iname in reduction_expr.inames:
            if iname.startswith("@"):
                new_iname = kernel.make_unique_var_name(iname[1:]+"_"+name_base,
                        newly_created_vars)

                old_inames.append(iname.lstrip("@"))
                new_inames.append(new_iname)
                newly_created_vars.add(new_iname)
                new_red_inames.append(new_iname)
                did_something = True
            else:
                new_red_inames.append(iname)

        if did_something:
            from loopy.symbolic import SubstitutionMapper
            from pymbolic.mapper.substitutor import make_subst_func
            from pymbolic import var

            subst_dict = dict(
                    (old_iname, var(new_iname))
                    for old_iname, new_iname in zip(
                        reduction_expr.untagged_inames, new_red_inames))
            subst_map = SubstitutionMapper(make_subst_func(subst_dict))

            child = subst_map(child)

        from loopy.symbolic import Reduction
        return Reduction(
                operation=reduction_expr.operation,
                inames=tuple(new_red_inames),
                expr=child)

    # }}}

    from loopy.symbolic import ReductionCallbackMapper
    from loopy.isl_helpers import duplicate_axes

    new_domains = kernel.domains
    new_insns = []

    new_iname_to_tag = kernel.iname_to_tag.copy()

    for insn in kernel.instructions:
        old_inames = []
        new_inames = []
        name_base = insn.id

        new_insns.append(insn.copy(
            expression=ReductionCallbackMapper(duplicate_reduction_inames)
            (insn.expression)))

        for old, new in zip(old_inames, new_inames):
            new_domains = duplicate_axes(new_domains, [old], [new])
            if old in kernel.iname_to_tag:
                new_iname_to_tag[new] = kernel.iname_to_tag[old]

    new_substs = {}
    for sub_name, sub_rule in kernel.substitutions.iteritems():
        old_inames = []
        new_inames = []
        name_base = sub_name

        new_substs[sub_name] = sub_rule.copy(
                expression=ReductionCallbackMapper(duplicate_reduction_inames)
                (sub_rule.expression))

        for old, new in zip(old_inames, new_inames):
            new_domains = duplicate_axes(new_domains, [old], [new])
            if old in kernel.iname_to_tag:
                new_iname_to_tag[new] = kernel.iname_to_tag[old]

    return kernel.copy(
            instructions=new_insns,
            substitutions=new_substs,
            domains=new_domains,
            iname_to_tag=new_iname_to_tag)

# }}}

# {{{ duplicate inames

def duplicate_inames(knl):
    new_insns = []
    new_domains = knl.domains
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
                new_iname_to_tag[new_iname] = new_tag

            newly_created_vars.update(new_inames)

            from loopy.isl_helpers import duplicate_axes
            new_domains = duplicate_axes(new_domains, inames_to_duplicate, new_inames)

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
            domains=new_domains,
            iname_to_tag=new_iname_to_tag)
# }}}

# {{{ kernel creation top-level

def make_kernel(*args, **kwargs):
    """Second pass of kernel creation. Think about requests for iname duplication
    and temporary variable creation.
    """

    from loopy.kernel import LoopKernel
    knl = LoopKernel(*args, **kwargs)

    from loopy import tag_dimensions
    knl = tag_dimensions(
            knl.copy(iname_to_tag_requests=None),
            knl.iname_to_tag_requests).copy(
                    iname_to_tag_requests=[])

    check_for_nonexistent_iname_deps(knl)

    knl = create_temporaries(knl)
    knl = duplicate_reduction_inames(knl)
    knl = duplicate_inames(knl)

    # -------------------------------------------------------------------------
    # Ordering dependency:
    # -------------------------------------------------------------------------
    # Must duplicate inames before expanding CSEs, otherwise inames within the
    # scope of duplication might be CSE'd out to a different instruction and
    # never be found by duplication.
    # -------------------------------------------------------------------------

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

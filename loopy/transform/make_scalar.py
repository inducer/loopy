from pymbolic.primitives import Variable
from loopy.symbolic import (RuleAwareIdentityMapper, SubstitutionRuleMappingContext)
from loopy.kernel.data import ValueArg
from loopy.transform.iname import remove_unused_inames


class ScalarChanger(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, var_name):
        self.var_name = var_name
        super().__init__(rule_mapping_context)

    def map_subscript(self, expr, expn_state):
        if expr.aggregate.name == self.var_name:
            return Variable(self.var_name)

        return super().map_subscript(expr, expn_state)


def make_scalar(kernel, var_name):
    rule_mapping_context = SubstitutionRuleMappingContext(kernel.substitutions,
            kernel.get_var_name_generator())

    kernel = ScalarChanger(rule_mapping_context, var_name).map_kernel(kernel)

    new_args = [ValueArg(arg.name, arg.dtype, target=arg.target,
        is_output_only=arg.is_output_only) if arg.name == var_name else arg for
        arg in kernel.args]
    new_temps = dict((tv.name, tv.copy(shape=(), dim_tags=None))
            if tv.name == var_name else (tv.name, tv) for tv in
            kernel.temporary_variables.values())

    return kernel.copy(args=new_args, temporary_variables=new_temps)


def remove_invariant_inames(kernel):
    inames_used = set()
    untagged_inames = (
            kernel.all_inames() - frozenset(kernel.iname_to_tags.keys()))
    for insn in kernel.instructions:
        for iname in ((insn.read_dependency_names()
            | insn.write_dependency_names())
        & untagged_inames):
            inames_used.add(iname)

    removable_inames = untagged_inames - inames_used

    new_insns = [insn.copy(within_inames=insn.within_inames-removable_inames)
            for insn in kernel.instructions]

    return remove_unused_inames(kernel.copy(instructions=new_insns),
            removable_inames)

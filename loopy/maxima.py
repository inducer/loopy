"""Export to maxima."""

from __future__ import division
from pymbolic.maxima import MaximaStringifyMapper as MaximaStringifyMapperBase

class MaximaStringifyMapper(MaximaStringifyMapperBase):
    def map_subscript(self, expr, enclosing_prec):
        res = self.rec(expr.aggregate, enclosing_prec)
        idx = expr.index
        if not isinstance(idx, tuple):
            idx = (idx,)
        for i in idx:
            if isinstance(i, int):
                res += "_%d" % i

        return res

def get_loopy_instructions_as_maxima(kernel, prefix):
    """Sample use for code comparison::

        load("knl-optFalse.mac");
        load("knl-optTrue.mac");

        vname: bessel_j_8;

        un_name : concat(''un_, vname);
        opt_name : concat(''opt_, vname);

        print(ratsimp(ev(un_name - opt_name)));
    """
    from loopy.preprocess import add_boostability_and_automatic_dependencies
    kernel = add_boostability_and_automatic_dependencies(kernel)

    my_variable_names = (
            insn.get_assignee_var_name() for insn in kernel.instructions)

    from pymbolic import var
    subst_dict = dict(
            (vn, var(prefix+vn)) for vn in my_variable_names)

    mstr = MaximaStringifyMapper()
    from loopy.symbolic import SubstitutionMapper
    from pymbolic.mapper.substitutor import make_subst_func
    substitute = SubstitutionMapper(make_subst_func(subst_dict))

    result = ["ratprint:false;"]

    written_insn_ids = set()

    from loopy.kernel import Instruction

    def write_insn(insn):
        if not isinstance(insn, Instruction):
            insn = kernel.id_to_insn[insn]

        for dep in insn.insn_deps:
            if dep not in written_insn_ids:
                write_insn(dep)

        result.append("%s%s : %s;" % (
            prefix, insn.get_assignee_var_name(),
            mstr(substitute(insn.expression))))

        written_insn_ids.add(insn.id)

    for insn in kernel.instructions:
        if insn.id not in written_insn_ids:
            write_insn(insn)

    return "\n".join(result)

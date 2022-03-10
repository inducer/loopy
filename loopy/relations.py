from kanren import Relation, facts


def get_inameo(kernel):
    inameo = Relation()
    for iname in kernel.all_inames():
        facts(inameo, (iname,))
    return inameo


def get_argo(kernel):
    argo = Relation()
    for arg in kernel.args:
        facts(argo, (arg.name,))

    return argo


def get_tempo(kernel):
    tempo = Relation()
    for tv in kernel.temporary_variables:
        facts(tempo, (tv,))

    return tempo


def get_insno(kernel):
    insno = Relation()
    for insn in kernel.instructions:
        facts(insno, (insn.id,))

    return insno


def get_taggedo(kernel):
    taggedo = Relation()

    for arg_name, arg in kernel.arg_dict.items():
        for tag in arg.tags:
            facts(taggedo, (arg_name, tag))

    for iname_name, iname in kernel.inames.items():
        for tag in iname.tags:
            facts(taggedo, (iname_name, tag))

    for insn in kernel.instructions:
        for tag in insn.tags:
            facts(taggedo, (insn.id, tag))

    return taggedo


def get_taggedo_of_type(kernel, tag_type):
    taggedo = Relation()

    for arg_name, arg in kernel.arg_dict.items():
        for tag in arg.tags_of_type(tag_type):
            facts(taggedo, (arg_name, tag))

    for iname_name, iname in kernel.inames.items():
        for tag in iname.tags_of_type(tag_type):
            facts(taggedo, (iname_name, tag))

    for insn in kernel.instructions:
        for tag in insn.tags_of_type(tag_type):
            facts(taggedo, (insn.id, tag))

    return taggedo


def get_producero(kernel):
    producero = Relation()

    for insn in kernel.instructions:
        for var in insn.assignee_var_names():
            facts(producero, (insn.id, var))

    return producero


def get_consumero(kernel):
    consumero = Relation()

    for insn in kernel.instructions:
        for var in insn.read_dependency_names():
            facts(consumero, (insn.id, var))

    return consumero


def get_withino(kernel):
    withino = Relation()

    for insn in kernel.instructions:
        facts(withino, (insn.id, insn.within_inames))

    return withino


def get_reduce_insno(kernel):
    reduce_insno = Relation()

    for insn in kernel.instructions:
        if insn.reduction_inames():
            facts(reduce_insno, (insn.id,))

    return reduce_insno


def get_reduce_inameo(kernel):
    from functools import reduce
    reduce_inameo = Relation()

    for iname in reduce(frozenset.union,
                        (insn.reduction_inames()
                         for insn in kernel.instructions),
                        frozenset()):
        facts(reduce_inameo, (iname,))

    return reduce_inameo

# vim: fdm=marker

import islpy as isl

from loopy.symbolic import BatchedAccessMapMapper
from loopy import LoopKernel
from loopy import InstructionBase

from itertools import product
from functools import reduce
from typing import Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class HappensBefore: 
    """A class representing a "happens-before" relationship between two
    statements found in a :class:`loopy.LoopKernel`. Used to validate that a
    given kernel transformation respects the data dependencies in a given
    program.

    .. attribute:: happens_before
        The :attr:`id` of a :class:`loopy.InstructionBase` that depends on the
        current :class:`loopy.InstructionBase` instance.

    .. attribute:: variable_name
        The name of the variable in a program that is causing the dependency. 

    .. attribute:: relation
        An :class:`isl.Map` representing the data dependency. The input of the
        map is an iname tuple and the output of the map is a set of iname tuples
        that must execute after the input.
    """
    
    happens_before: str
    variable_name: Optional[str]
    relation: isl.Map

# TODO Do we really need this?
@dataclass(frozen=True)
class _AccessRelation:
    """A class that stores information about a particular array access in a
    program.
    .. attribute:: id
        The instruction id of the statement the access relation is representing.

    .. attribute:: variable_name
        The memory location the access relation is representing.

    .. attribute:: relation
        An :class:`isl.Map` object representing the memory access. The access
        relation is a map from the loop domain to the set of valid array
        indices.

    .. attribute:: access_type
        An :class:`Enum` object representing the type of memory access the
        statement is making. The type of memory access is either a read or a
        write.
    """

    id: str
    variable_name: str
    relation: isl.Map

def generate_dependency_relations(knl: LoopKernel) -> list[HappensBefore]:
    """Generates :class:`isl.Map` objects representing the data dependencies between
    statements in a loopy program. The :class:`isl.Map` objects are stored in a
    :class:`loopy.Dependency.HappensBefore` object along with the dependee id,
    variable name, and dependency type.
    
    :arg knl: A :class:`loopy.LoopKernel` containing the instructions we wish to
    find data dependencies between.

    :returns: Three lists containing :class:`loopy.Dependency.HappensBefore`
    objects describing the data dependencies.
    """
    bmap: BatchedAccessMapMapper = BatchedAccessMapMapper(knl,
                                      knl.all_variable_names())
    for insn in knl.instructions:
        bmap(insn.assignee, insn.within_inames)
        bmap(insn.expression, insn.within_inames)

    def get_map(var: str, insn: InstructionBase) -> isl.Map: 
        return bmap.access_maps[var][insn.within_inames]

    def read_variables(insn: InstructionBase) -> frozenset[str]:
        return insn.read_dependency_names() - insn.within_inames

    def write_variables(insn: InstructionBase) -> frozenset[str]: 
        return insn.write_dependency_names() - insn.within_inames

    def variable_list(insn: InstructionBase) -> frozenset[str]:
        return read_variables(insn) | write_variables(insn)

    def dependency_relation(x: isl.Map, y:isl.Map) -> isl.Map:
        dependency: isl.Map = x.apply_range(y.reverse())
        diagonal: isl.Map = dependency.identity(dependency.get_space()) 
        dependency -= diagonal

        return dependency

    # TODO can we reduce this code even further by not computing access
    # relations before we begin computing dependencies?
    accesses: list[_AccessRelation] = [_AccessRelation(insn.id, var,
                                                       get_map(var, insn)) 
                                       for insn in knl.instructions
                                       for var in variable_list(insn)]

    dependencies: list[HappensBefore] = [HappensBefore(dependent.id,
                                   dependee.variable_name,
                                   dependency_relation(dependent.relation,
                                                       dependee.relation))
                         for dependee, dependent in product(*[accesses,accesses])
                         if dependent.variable_name == dependee.variable_name]

    return dependencies  

def generate_execution_order(knl: LoopKernel) -> frozenset[isl.Map]:
    """Generate the "happens-before" execution order that *must* be respected by
    any transformation. Calls :function:`generate_dependency_relations` to get
    the information needed to compute the execution order.

    :arg knl: A :class:`loopy.LoopKernel` containing the instructions for which
    to generate a "happens-before" execution order.

    :returns: A :class:`frozenset` of :class:`isl.Map` representing the
    execution required by the dependencies in a loopy program.
    """

    dependencies: list[HappensBefore] = generate_dependency_relations(knl)
    execution_order: frozenset[isl.Map] = frozenset()
    for insn in knl.instructions:
        # TODO replace lexicographical domain with existing happens before
        domain: isl.BasicSet = knl.get_inames_domain(insn.within_inames)
        insn_order: isl.Map = domain.lex_lt_set(domain) & \
                              reduce(lambda x, y: x | y, [dep.relation for dep in
                                                          dependencies])
        execution_order: frozenset[isl.Map] = execution_order | frozenset({insn_order})

    return execution_order 

def verify_execution_order(knl: LoopKernel, existing_happens_before: isl.Map):
    """Verify that a given transformation respects the dependencies in a
    :class:`loopy.LoopKernel` program. Calls
    :function:`generate_execution_order` to generate the "happens-before" for
    each iname domain that *must* be respected in order for a transformation to
    be valid.

    :returns: True or false depending on whether the provided execution order
    respects the dependencies in the loopy program.
    """
    pass

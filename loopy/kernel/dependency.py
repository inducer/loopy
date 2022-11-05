import islpy as isl

from loopy.symbolic import BatchedAccessMapMapper
from loopy import LoopKernel
from loopy import InstructionBase

from functools import reduce
from typing import Optional
from dataclasses import dataclass
from enum import Enum

class DependencyType(Enum):
    """An enumeration of the types of data dependencies found in a program.
    """
    WRITE_AFTER_READ  = 0
    READ_AFTER_WRITE  = 1
    WRITE_AFTER_WRITE = 2

class AccessType(Enum):
    """An enumeration of the types of accesses made by statements in a program.
    """
    READ  = 0
    WRITE = 1

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

    .. attribute:: dependency_type
        A :class:`DependencyType` of :class:`Enum` representing the dependency
        type (write-read, read-write, write-write). 
    """
    
    happens_before: str
    variable_name: Optional[str]
    relation: isl.Map
    dependency_type: Optional[DependencyType]

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
    access_type: AccessType

def generate_dependency_relations(knl: LoopKernel) \
        -> tuple[list[HappensBefore], list[HappensBefore], list[HappensBefore]]:
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

    def read_var_list(insn: InstructionBase) -> frozenset[str]:
        return insn.read_dependency_names() - insn.within_inames

    def write_var_list(insn: InstructionBase) -> frozenset[str]: 
        return insn.write_dependency_names() - insn.within_inames

    def get_dependency_relation(x: isl.Map, y:isl.Map) -> isl.Map:
        dependency: isl.Map = x.apply_range(y.reverse())
        diagonal: isl.Map = dependency.identity(dependency.get_space()) 
        dependency -= diagonal

        return dependency

    read_maps: list[_AccessRelation] = [_AccessRelation(insn.id, var, 
                                                      get_map(var, insn),
                                                      AccessType.READ)
                 for insn in knl.instructions
                 for var in read_var_list(insn)]
    write_maps: list[_AccessRelation] = [_AccessRelation(insn.id, var, 
                                                      get_map(var, insn),
                                                      AccessType.WRITE)
                 for insn in knl.instructions
                 for var in write_var_list(insn)]

    write_read: list[HappensBefore] = [HappensBefore(read.id,
                                        write.variable_name,
                                        get_dependency_relation(write.relation,
                                                                read.relation),
                                        DependencyType.WRITE_AFTER_READ)
                                       for write in write_maps
                                       for read in read_maps
                                       if write.variable_name == read.variable_name]
    read_write: list[HappensBefore] = [HappensBefore(write.id,
                                        read.variable_name,
                                        get_dependency_relation(read.relation,
                                                                write.relation),
                                        DependencyType.READ_AFTER_WRITE)
                                       for read in read_maps
                                       for write in write_maps
                                       if read.variable_name == write.variable_name]
    write_write: list[HappensBefore] = [HappensBefore(write2.id,
                                         write1.variable_name,
                                         get_dependency_relation(write1.relation,
                                         write2.relation),
                                         DependencyType.WRITE_AFTER_WRITE)
                                        for write1 in write_maps
                                        for write2 in write_maps
                                        if write1.variable_name == write2.variable_name]

    return write_read, read_write, write_write

def generate_execution_order(knl: LoopKernel) -> frozenset[isl.Map]:
    """Generate the "happens-before" execution order that *must* be respected by
    any transformation. Calls :function:`generate_dependency_relations` to get
    the information needed to compute the execution order.

    :arg knl: A :class:`loopy.LoopKernel` containing the instructions for which
    to generate a "happens-before" execution order.

    :returns: A :class:`frozenset` of :class:`isl.Map` representing the
    execution required by the dependencies in a loopy program.
    """

    write_read, read_write, write_write = generate_dependency_relations(knl)
    
    execution_order: frozenset[isl.Map] = frozenset()

    for insn in knl.instructions:
        domain: isl.BasicSet = knl.get_inames_domain(insn.within_inames)
        insn_order: isl.Map = domain.lex_lt_set(domain)

        union_of_dependencies: isl.Map = reduce(lambda x, 
                                                y: x.relation | y.relation,
                                                write_read)
        union_of_dependencies |= reduce(lambda x, y: x.relation | y.relation,
                                        read_write)
        union_of_dependencies |= reduce(lambda x, y: x.relation | y.relation,
                                        write_write)
        
        insn_order: isl.Map = insn_order & union_of_dependencies
        execution_order = execution_order | frozenset({insn_order})

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

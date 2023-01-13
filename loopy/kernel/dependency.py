import pymbolic.primitives as p

from dataclasses import dataclass
from islpy import Map
from typing import FrozenSet, Optional, List

from loopy import LoopKernel
from loopy import InstructionBase
from loopy.symbolic import WalkMapper

@dataclass(frozen=True)
class HappensAfter:
    variable_name: Optional[str]
    instances_rel: Optional[Map]

class AccessMapMapper(WalkMapper):
    """
    TODO Update this documentation so it reflects proper formatting

    Used instead of BatchedAccessMapMapper to get single access maps for each
    instruction. 
    """

    def __init__(self, kernel, var_names):
        self.kernel = kernel
        self._var_names = var_names

        from collections import defaultdict
        self.access_maps = defaultdict(lambda:
                           defaultdict(lambda:
                           defaultdict(lambda: None)))
        
        super.__init__()

    def map_subscript(self, expr, inames, insn_id):    

        domain = self.kernel.get_inames_domain(inames)

        # why do we need this?
        WalkMapper.map_subscript(self, expr, inames)

        assert isinstance(expr.aggregate, p.Variable)

        if expr.aggregate.name not in self._var_names:
            return

        arg_name = expr.aggregate.name
        subscript = expr.index_tuple

        descriptor = self.kernel.get_var_descriptor(arg_name)

        from loopy.diagnostic import UnableToDetermineAccessRangeError
        from loopy.symbolic import get_access_map

        try:
            access_map = get_access_map(domain, subscript)
        except UnableToDetermineAccessRangeError:
            return
        
        if self.access_maps[insn_id][arg_name][inames] is None:
            self.access_maps[insn_id][arg_name][inames] = access_map

def compute_happens_after(knl: LoopKernel) -> LoopKernel: 
    """
    TODO Update documentation to reflect the proper format.

    Determine dependency relations that exist between instructions. Similar to
    apply_single_writer_dependency_heuristic. Extremely rough draft.
    """
    writer_map = knl.writer_map()
    variables = knl.all_variable_names - knl.inames.keys()

    amap = AccessMapMapper(knl, variables)

    dep_map = {
        insn.id: insn.read_dependency_names() - insn.within_inames
        for insn in knl.instructions
    }

    new_insns = []
    for insn in knl.instructions:
        current_insn = insn.id
        inames = insn.within_inames

        new_happens_after = []
        for var in dep_map[insn.id]:
            for writer in (writer_map.get(var, set()) - { current_insn }):

                # get relation for current instruction and a write instruction
                cur_relation = amap.access_maps[current_insn][var][inames] 
                write_relation = amap.access_maps[writer][var][inames]

                # compute the dependency relation
                dep_relation = cur_relation.apply_range(write_relation.reverse())

                # create the mapping from writer -> (variable, dependency rel'n)
                happens_after = HappensAfter(var, dep_relation)
                happens_after_mapping = { writer: happens_after }

                # add to the new list of dependencies
                new_happens_after |= happens_after_mapping
        
        # update happens_after of our current instruction with the mapping 
        insn = insn.copy(happens_after=new_happens_after)
        new_insns.append(insn)

    # return the kernel with the new instructions
    return knl.copy(instructions=new_insns)

def add_lexicographic_happens_after(knl: LoopKernel) -> None:
    pass


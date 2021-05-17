import pymbolic.primitives as prim
import loopy.schedule as schedule
from loopy.diagnostic import LoopyError
from typing import List, Union, Any, Optional, Tuple
from dataclasses import dataclass, field
import islpy as isl
from islpy import dim_type
from functools import reduce


# {{{ LoopKernel.schedule a tree

class ScheduleNode:
    """
    Abstract class for a schedule node in a class:`~loopy.LoopKernel`.
    """
    pass


@dataclass
class RunInstruction(ScheduleNode):
    insn_id: str

    mapper_method: str = field(default="map_run_instruction", repr=False, init=False)


@dataclass
class Barrier(ScheduleNode):
    """
    .. attribute:: comment

        A plain-text comment explaining why the barrier was inserted.

    .. attribute:: synchronization_kind

        ``"local"`` or ``"global"``

    .. attribute:: mem_kind

        ``"local"`` or ``"global"``

    .. attribute:: originating_insn_id
    """
    comment: str
    synchronization_kind: str
    originating_insn_id: Optional[str]

    mapper_method: str = field(default="map_barrier", repr=False, init=False)


@dataclass
class InstructionBlock(ScheduleNode):
    """
    List of instruction ids that are to be executed in sequence. An instruction
    block cannot contain other blocks or loops.

    .. attribute:: children

        A list of instruction ids contained in the block.
    """
    children: List[Union[Barrier, RunInstruction]]

    mapper_method: str = field(default="map_instruction_block", repr=False,
                               init=False)


@dataclass
class Loop(ScheduleNode):
    """
    A loop with the induction variable *iname*.
    """
    iname: str
    children: List[Union[InstructionBlock, "Loop"]]

    mapper_method: str = field(default="map_loop", repr=False, init=False)


@dataclass
class Function(ScheduleNode):
    """
    A function definition.

    .. attribute:: name

        An instance of :class:`str`

    .. attribute:: extra_args

    .. attribute:: extra_inames
    """
    name: str
    extra_args: List[Any]
    extra_inames: List[str]
    children: List[Union[InstructionBlock, Loop]]

    mapper_method: str = field(default="map_function", repr=False, init=False)


class For(Loop):
    iname: str
    upper_bound: Union[int, prim.Expression]
    lower_bound: Union[int, prim.Expression]
    step: int

    mapper_method: str = field(default="map_resolved_for", repr=False, init=False)


class If(ScheduleNode):
    condition: Union[int, bool, prim.Expression]


@dataclass
class Schedule(ScheduleNode):
    """
    Top-level schedule description.
    """
    children: List[Union[Loop, InstructionBlock, Function]]

    mapper_method: str = field(default="map_schedule", repr=False, init=False)


@dataclass
class ScheduleTreeBuilder:
    """
    A builder for :class:`Schedule`.
    """

    schedule: Schedule
    _build_stack: List[ScheduleNode]

    @staticmethod
    def new():
        sched = Schedule([])
        return ScheduleTreeBuilder(sched, [sched])

    @property
    def current_node(self):
        return self._build_stack[-1]

    def make_current_node(self, node):
        self._build_stack.append(node)

    def make_and_enter_function(self, name, extra_args, extra_inames):
        assert isinstance(self.current_node, Schedule)
        new_function = Function(name, extra_args, extra_inames, [])
        self.current_node.children.append(new_function)
        self.make_current_node(new_function)

    def make_and_enter_instruction_block(self):
        assert isinstance(self.current_node, (Function, Loop, Schedule))
        new_block = InstructionBlock([])
        self.current_node.children.append(new_block)
        self.make_current_node(new_block)

    def make_and_enter_loop(self, iname):
        assert isinstance(self.current_node, (Schedule, Function, Loop))
        new_loop = Loop(iname, [])
        self.current_node.children.append(new_loop)
        self.make_current_node(new_loop)

    def add_run_instruction(self, insn_id):
        if not isinstance(self.current_node, InstructionBlock):
            self.make_and_enter_instruction_block()

        self.current_node.children.append(RunInstruction(insn_id))

    def add_barrier(self, comment, kind, insn_id):
        if not isinstance(self.current_node, InstructionBlock):
            self.make_instruction_block()

        self.current_node.children.append(Barrier(comment, kind, insn_id))

    def exit_function(self):
        if isinstance(self.current_node, InstructionBlock):
            self._build_stack.pop()
        assert isinstance(self.current_node, Function)
        return self._build_stack.pop()

    def exit_loop(self):
        if isinstance(self.current_node, InstructionBlock):
            self._build_stack.pop()
        assert isinstance(self.current_node, Loop)
        return self._build_stack.pop()

    def exit(self):
        if isinstance(self.current_node, InstructionBlock):
            self._build_stack.pop()
        assert isinstance(self.current_node, Schedule)
        return self._build_stack.pop()


def make_schedule_tree(kernel):
    # bob: the schedule builder
    bob = ScheduleTreeBuilder.new()

    for sched_item in kernel.schedule:
        if isinstance(sched_item, schedule.CallKernel):
            bob.make_and_enter_function(sched_item.kernel_name,
                                        sched_item.extra_args,
                                        sched_item.extra_inames)
        elif isinstance(sched_item, schedule.ReturnFromKernel):
            fn = bob.exit_function()
            assert fn.name == sched_item.kernel_name
        elif isinstance(sched_item, schedule.EnterLoop):
            bob.make_and_enter_loop(sched_item.iname)
        elif isinstance(sched_item, schedule.LeaveLoop):
            loop = bob.exit_loop()
            assert loop.iname == sched_item.iname
        elif isinstance(sched_item, schedule.RunInstruction):
            bob.add_run_instruction(sched_item.insn_id)
        elif isinstance(sched_item, schedule.Barrier):
            bob.add_barrier(sched_item.comment,
                            sched_item.synchronization_kind,
                            sched_item.originating_insn_id)
        else:
            raise NotImplementedError(type(sched_item))

    return bob.exit()

# }}}


class Mapper:
    def __call__(self, expr, *args, **kwargs):
        try:
            method = getattr(self, expr.mapper_method)
        except AttributeError:
            raise LoopyError(f"{type(self)} cannot handle expressions of"
                             f" type {type(expr)}.")

        return method(expr, *args, **kwargs)

    rec = __call__


class IdentityMapper(Mapper):
    def map_schedule(self, expr, *args, **kwargs):
        return Schedule([self.rec(child, *args, **kwargs)
                         for child in expr.children])

    def map_instruction_block(self, expr, *args, **kwargs):
        return InstructionBlock([self.rec(child, *args, **kwargs)
                                 for child in expr.children])

    def map_function(self, expr, *args, **kwargs):
        return Function(expr.name,
                        expr.extra_args,
                        expr.extra_inames,
                        [self.rec(child, *args, **kwargs)
                         for child in expr.children])

    def map_loop(self, expr, *args, **kwargs):
        return Loop(expr.iname,
                    [self.rec(child, *args, **kwargs)
                     for child in expr.children])

    def map_barrier(self, expr, *args, **kwargs):
        return Barrier(expr.comment, expr.synchronization_kind,
                       expr.originating_insn_id)

    def map_run_instruction(self, expr, *args, **kwargs):
        return RunInstruction(expr.insn_id)


class CombineMapper(Mapper):
    def combine(self, values):
        raise NotImplementedError

    def map_schedule(self, expr, *args, **kwargs):
        return self.combine([self.rec(child, *args, **kwargs)
                             for child in expr.children])

    def map_instruction_block(self, expr, *args, **kwargs):
        return self.combine([self.rec(child, *args, **kwargs)
                             for child in expr.children])

    def map_function(self, expr, *args, **kwargs):
        return self.combine([self.rec(child, *args, **kwargs)
                             for child in expr.children])

    def map_loop(self, expr, *args, **kwargs):
        return self.combine([self.rec(child, *args, **kwargs)
                             for child in expr.children])

    def map_barrier(self, expr, *args, **kwargs):
        raise NotImplementedError

    def map_run_instruction(self, expr, *args, **kwargs):
        raise NotImplementedError


class StringifyMapper(CombineMapper):
    SHIFTWIDTH = 2

    def __init__(self, kernel):
        self.kernel = kernel

    def combine(self, values):
        return "\n".join(values)

    def _indent(self, level):
        return level*self.SHIFTWIDTH*" "

    def map_function(self, expr, level=0):
        return self.combine([(f"{self._indent(level)}CALL KERNEL {expr.name}("
                              f"extra_args={expr.extra_args}, "
                              f"extra_inames={expr.extra_inames})"),
                             super().map_function(expr, level+1),
                             f"{self._indent(level)}RETURN FROM KERNEL {expr.name}"])

    def map_run_instruction(self, expr, level=0):
        from loopy.schedule import format_insn
        return (f"{self._indent(level)}"
                f"{format_insn(self.kernel, expr.insn_id)}")

    def map_barrier(self, expr, level=0):
        return (f"{self._indent(level)}... {expr.kind[0]}barrier")

    def map_loop(self, expr, level=0):
        return self.combine([f"{self._indent(level)}for {expr.iname}",
                             super().map_function(expr, level+1),
                             f"{self._indent(level)}end {expr.iname}"])


def _align_and_intersect(d1, d2):
    d1, d2 = isl.align_two(d1, d2)
    return d1 & d2


def _wrap_in_if(cond, nodes):
    if cond.is_universe():
        return nodes
    else:
        return If(cond, nodes)


@dataclass(frozen=True)
class PredicateInsertionContext:
    implemented_domain: isl.BasicSet
    gsize: Optional[Tuple[isl.PwAff, ...]] = None
    lsize: Optional[Tuple[isl.PwAff, ...]] = None

    def copy(self, *, implemented_domain=None, gsize=None, lsize=None):
        if implemented_domain is None:
            implemented_domain = self.implemented_domain

        if gsize is None:
            gsize = self.gsize

        if lsize is None:
            lsize = self.lsize

        return PredicateInsertionContext(implemented_domain, gsize, lsize)


class PredicateInsertion(IdentityMapper):
    def __init__(self, kernel):
        self.kernel = kernel

    def map_schedule(self, expr):
        universe = isl.BasicSet(isl.Space.create_from_names(self.kernel.isl_context,
                                                            []))

        return super().map_schedule(expr, PredicateInsertionContext(universe))

    def map_function(self, expr, context):
        # get the implemented domain for the insn ids in this kernel
        # Shouldn't be difficult to write a combine mapper for it.
        gsize, lsize = self.kernel.get_grid_sizes_for_insns_ids(
            gather_insn_ids_for_kernel(expr.name))
        return super().map_function(expr, context.copy(gsize=gsize, lsize=lsize))

    def map_run_instruction(self, expr, context):
        return expr

    def map_barrier(self, expr, context):
        return expr

    def map_instruction_block(self, expr, context):
        if all(isinstance(child, RunInstruction) for child in expr.children):
            # need to add a predicate for the hardware axes usage.
            assert len({self.kernel.id_to_insn[child.insn_id].within_inames
                        for child in expr.children}) == 1
            inames = self.kernel.id_to_insn[expr.children[0].insn_id].within_inames
            hw_inames = inames - set(context.implemented_domain.get_var_dict())
            if hw_inames:
                raise NotImplementedError

            return InstructionBlock([self.rec(child, context)
                                     for child in expr.children])
        else:
            assert all(isinstance(child, Barrier) for child in expr.childre)
            return InstructionBlock([self.rec(child, context)
                                     for child in expr.children])

    def map_loop(self, expr, context):
        from loopy.symbolic import aff_to_expr, set_to_cond_expr

        implemented_domain = context.implemented_domain
        assert implemented_domain.dim(dim_type.set) == 0

        domain = self.kernel.get_iname_domain(expr.iname)

        # {{{ make already implemented loops as parallel; project out inner loops

        for set_dim in domain.get_var_names(dim_type.set):
            dt, pos = domain.get_var_dict()[set_dim]
            assert dt == dim_type.set

            if set_dim in implemented_domain.get_var_dict():
                # make outer loop's iname a param
                domain = domain.move_dims(dim_type.param,
                                          domain.dim(dim_type.param),
                                          dt, pos, 1)
            elif set_dim != expr.name:
                domain = domain.project_out(dt, pos, 1)
            else:
                pass

        # }}}

        assert domain.dim(dim_type.set) == 1

        domain, implemented_domain = isl.align_two(domain,
                                                   implemented_domain)
        domain = domain.gist(implemented_domain)

        downstream_domain = _align_and_intersect(domain
                                                 .move_dims(dim_type.param,
                                                            domain.dim(dim_type
                                                                       .param),
                                                            dim_type.set,
                                                            0, 1),
                                                 implemented_domain)

        outer_condition = isl.align_space(domain.project_out(dim_type.set, 0),
                                          downstream_domain).gist(downstream_domain)

        min_affs = [aff for set, aff in domain.dim_min(0)]
        max_affs = [aff for set, aff in domain.dim_max(0)]
        lower_bound = reduce(isl.Aff.min, min_affs, min_affs[0])
        upper_bound = reduce(isl.Aff.max, max_affs, max_affs[0])

        inner_condition = domain.affine_hull()
        step = 1  # TODO: from inner_condition try to guess the step

        children = [self.rec(child, (context
                                     .copy(implemented_domain=downstream_domain)))
                    for child in expr.children]

        return _wrap_in_if(outer_condition,
                           For(aff_to_expr(lower_bound),
                               aff_to_expr(upper_bound),
                               step,
                               _wrap_in_if(set_to_cond_expr(inner_condition),
                                           children)))


class InstructionGatherer(CombineMapper):
    """
    Mapper to gather all insn ids a :class:`Function`.
    """
    def __init__(self, function_name):
        self.function_name = function_name

    def combine(self, values):
        assert all(isinstance(value, list) for value in values)
        return sum(values, [])

    def map_function(self, expr):
        if expr.name == self.function_name:
            return super().map_function(expr)
        else:
            return []

    def map_run_instruction(self, expr):
        return [expr.insn_id]

    def map_barrier(self, expr):
        if expr.originating_insn_id is not None:
            return [expr.originating_insn_id]
        else:
            return []


def gather_insn_ids_for_kernel(schedule, kernel_name):
    return InstructionGatherer(kernel_name)(schedule)

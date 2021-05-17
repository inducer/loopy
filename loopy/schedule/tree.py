import loopy.schedule as schedule
from loopy.diagnostic import LoopyError
from typing import List, Union, Any, Optional
from dataclasses import dataclass, field


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

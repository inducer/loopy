__copyright__ = "Copyright (C) 2018 Kaushik Kulkarni"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import islpy as isl

from pytools import UniqueNameGenerator

from loopy.kernel import LoopKernel
from loopy.diagnostic import LoopyError
from loopy.kernel.instruction import (CallInstruction, MultiAssignmentBase,
        Assignment, CInstruction, _DataObliviousInstruction)
from loopy.symbolic import IdentityMapper, SubstitutionMapper, CombineMapper
from loopy.isl_helpers import simplify_via_aff
from loopy.kernel.function_interface import (
        CallableKernel, ScalarCallable)
from loopy.program import Program
from loopy.symbolic import SubArrayRef

__doc__ = """
.. currentmodule:: loopy

.. autofunction:: register_callable

.. autofunction:: merge
"""


def register_callable(translation_unit, function_identifier, callable_,
        redefining_not_ok=True):
    """
    :param translation_unit: A :class:`loopy.Program`.
    :param callable_: A :class:`loopy.InKernelCallable`.
    """

    if isinstance(callable_, LoopKernel):
        callable_ = CallableKernel(callable_)

    from loopy.kernel.function_interface import InKernelCallable
    assert isinstance(callable_, InKernelCallable)

    if (function_identifier in translation_unit.callables_table) and (
            translation_unit.callables_table[function_identifier] != callable_
            and redefining_not_ok):
        raise LoopyError("Redifining function identifier not allowed. Set the"
                " option 'redefining_not_ok=False' to bypass this error.")

    callables = translation_unit.callables_table.copy()
    callables[function_identifier] = callable_

    return translation_unit.copy(
            callables_table=callables)


def merge(translation_units):
    """
    :param translation_units: A list of :class:`loopy.Program`.

    :returns: An instance of :class:`loopy.Program` which contains all the
        callables from each of the *translation_units.
    """

    for i in range(1, len(translation_units)):
        if translation_units[i].target != translation_units[i-1].target:
            raise LoopyError("translation units to be merged should have the"
                             " same target.")

    # {{{ check for callable collision

    for i, prg_i in enumerate(translation_units):
        for prg_j in translation_units[i+1:]:
            for clbl_name in (set(prg_i.callables_table)
                              & set(prg_j.callables_table)):
                if (prg_i.callables_table[clbl_name]
                        != prg_j.callables_table[clbl_name]):
                    # FIXME: generate unique names + rename for the colliding
                    # callables
                    raise NotImplementedError("Translation units to be merged"
                                              " must have different callable names"
                                              " for now.")

    # }}}

    callables_table = {}
    for trans_unit in translation_units:
        callables_table.update(trans_unit.callables_table.copy())

    return Program(
            entrypoints=frozenset().union(*(
                t.entrypoints or frozenset() for t in translation_units)),
            callables_table=callables_table,
            target=translation_units[0].target)


# {{{ kernel inliner mapper

class KernelInliner(SubstitutionMapper):
    """Mapper to replace variables (indices, temporaries, arguments) in the
    callee kernel with variables in the caller kernel.

    :arg caller: the caller kernel
    :arg arg_map: dict of argument name to variables in caller
    :arg arg_dict: dict of argument name to arguments in callee
    """

    def __init__(self, subst_func, caller, arg_map, arg_dict):
        super().__init__(subst_func)
        self.caller = caller
        self.arg_map = arg_map
        self.arg_dict = arg_dict

    def map_subscript(self, expr):
        if expr.aggregate.name in self.arg_map:

            aggregate = self.subst_func(expr.aggregate)
            sar = self.arg_map[expr.aggregate.name]  # SubArrayRef in caller
            callee_arg = self.arg_dict[expr.aggregate.name]  # Arg in callee
            if aggregate.name in self.caller.arg_dict:
                caller_arg = self.caller.arg_dict[aggregate.name]  # Arg in caller
            else:
                caller_arg = self.caller.temporary_variables[aggregate.name]

            # Firstly, map inner inames to outer inames.
            outer_indices = self.map_tuple(expr.index_tuple)

            # Next, reshape to match dimension of outer arrays.
            # We can have e.g. A[3, 2] from outside and B[6] from inside
            from numbers import Integral
            if not all(isinstance(d, Integral) for d in callee_arg.shape):
                raise LoopyError(
                    "Argument: {} in callee kernel does not have "
                    "constant shape.".format(callee_arg))

            flatten_index = 0
            from loopy.symbolic import get_start_subscript_from_sar
            for i, idx in enumerate(get_start_subscript_from_sar(sar,
                    self.caller).index_tuple):
                flatten_index += idx*caller_arg.dim_tags[i].stride

            flatten_index += sum(
                idx * tag.stride
                for idx, tag in zip(outer_indices, callee_arg.dim_tags))

            from loopy.isl_helpers import simplify_via_aff
            flatten_index = simplify_via_aff(flatten_index)

            new_indices = []
            for dim_tag in caller_arg.dim_tags:
                ind = flatten_index // dim_tag.stride
                flatten_index -= (dim_tag.stride * ind)
                new_indices.append(ind)

            new_indices = tuple(simplify_via_aff(i) for i in new_indices)

            return aggregate.index(tuple(new_indices))
        else:
            return super().map_subscript(expr)

# }}}


# {{{ inlining of a single call instruction

def _inline_call_instruction(caller_kernel, callee_knl, instruction):
    """
    Returns a copy of *kernel* with the *instruction* in the *kernel*
    replaced by inlining :attr:`subkernel` within it.
    """
    callee_label = callee_knl.name[:4] + "_"

    # {{{ duplicate and rename inames

    vng = caller_kernel.get_var_name_generator()
    ing = caller_kernel.get_instruction_id_generator()
    dim_type = isl.dim_type.set

    iname_map = {}
    for iname in callee_knl.all_inames():
        iname_map[iname] = vng(callee_label+iname)

    new_domains = []
    new_iname_to_tags = caller_kernel.iname_to_tags.copy()

    # transferring iname tags info from the callee to the caller kernel
    for domain in callee_knl.domains:
        new_domain = domain.copy()
        for i in range(new_domain.n_dim()):
            iname = new_domain.get_dim_name(dim_type, i)

            if iname in callee_knl.iname_to_tags:
                new_iname_to_tags[iname_map[iname]] = (
                        callee_knl.iname_to_tags[iname])
            new_domain = new_domain.set_dim_name(
                dim_type, i, iname_map[iname])
        new_domains.append(new_domain)

    kernel = caller_kernel.copy(domains=caller_kernel.domains + new_domains,
            iname_to_tags=new_iname_to_tags)

    # }}}

    # {{{ rename temporaries

    temp_map = {}
    new_temps = kernel.temporary_variables.copy()
    for name, temp in callee_knl.temporary_variables.items():
        new_name = vng(callee_label+name)
        temp_map[name] = new_name
        new_temps[new_name] = temp.copy(name=new_name)

    kernel = kernel.copy(temporary_variables=new_temps)

    # }}}

    # {{{ match kernel arguments

    arg_map = {}  # callee arg name -> caller symbols (e.g. SubArrayRef)

    assignees = instruction.assignees  # writes
    parameters = instruction.expression.parameters  # reads

    # add keyword parameters
    from pymbolic.primitives import CallWithKwargs

    from loopy.kernel.function_interface import get_kw_pos_association
    kw_to_pos, pos_to_kw = get_kw_pos_association(callee_knl)
    if isinstance(instruction.expression, CallWithKwargs):
        kw_parameters = instruction.expression.kw_parameters
    else:
        kw_parameters = {}

    for kw, par in kw_parameters.items():
        arg_map[kw] = par

    for i, par in enumerate(parameters):
        arg_map[pos_to_kw[i]] = par

    for i, assignee in enumerate(assignees):
        arg_map[pos_to_kw[-i-1]] = assignee

    print(arg_map)

    # }}}

    # {{{ rewrite instructions

    import pymbolic.primitives as p
    from pymbolic.mapper.substitutor import make_subst_func

    var_map = {p.Variable(k): p.Variable(v)
                   for k, v in iname_map.items()}
    var_map.update({p.Variable(k): p.Variable(v)
                        for k, v in temp_map.items()})
    for k, v in arg_map.items():
        if isinstance(v, SubArrayRef):
            var_map[p.Variable(k)] = v.subscript.aggregate
        else:
            var_map[p.Variable(k)] = v

    subst_mapper = KernelInliner(
        make_subst_func(var_map), kernel, arg_map, callee_knl.arg_dict)

    insn_id = {}
    for insn in callee_knl.instructions:
        insn_id[insn.id] = ing(callee_label+insn.id)

    # {{{ root and leave instructions in callee kernel

    dep_map = callee_knl.recursive_insn_dep_map()
    # roots depend on nothing
    heads = {insn for insn, deps in dep_map.items() if not deps}
    # leaves have nothing that depends on them
    tails = set(dep_map.keys())
    for insn, deps in dep_map.items():
        tails = tails - deps

    # }}}

    # {{{ use NoOp to mark the start and end of callee kernel

    from loopy.kernel.instruction import NoOpInstruction

    noop_start = NoOpInstruction(
        id=ing(callee_label+"_start"),
        within_inames=instruction.within_inames,
        depends_on=instruction.depends_on
    )
    noop_end = NoOpInstruction(
        id=instruction.id,
        within_inames=instruction.within_inames,
        depends_on=frozenset(insn_id[insn] for insn in tails)
    )
    # }}}

    inner_insns = [noop_start]

    for insn in callee_knl.instructions:
        insn = insn.with_transformed_expressions(subst_mapper)
        within_inames = frozenset(map(iname_map.get, insn.within_inames))
        within_inames = within_inames | instruction.within_inames
        depends_on = frozenset(map(insn_id.get, insn.depends_on)) | (
                instruction.depends_on)
        if insn.id in heads:
            depends_on = depends_on | {noop_start.id}

        new_atomicity = tuple(
                type(atomicity)(var_map[p.Variable(atomicity.var_name)].name)
                for atomicity in insn.atomicity)

        if isinstance(insn, Assignment):
            insn = insn.copy(
                id=insn_id[insn.id],
                within_inames=within_inames,
                # TODO: probaby need to keep priority in callee kernel
                priority=instruction.priority,
                depends_on=depends_on,
                tags=insn.tags | instruction.tags,
                atomicity=new_atomicity
            )
        else:
            insn = insn.copy(
                id=insn_id[insn.id],
                within_inames=within_inames,
                # TODO: probaby need to keep priority in callee kernel
                priority=instruction.priority,
                depends_on=depends_on,
                tags=insn.tags | instruction.tags,
            )
        inner_insns.append(insn)

    inner_insns.append(noop_end)

    new_insns = []
    for insn in kernel.instructions:
        if insn == instruction:
            new_insns.extend(inner_insns)
        else:
            new_insns.append(insn)

    kernel = kernel.copy(instructions=new_insns)

    # }}}

    return kernel

# }}}


# {{{ inline callable kernel

def _inline_single_callable_kernel(caller_kernel, callee_kernel,
        callables_table):
    for insn in caller_kernel.instructions:
        if isinstance(insn, CallInstruction):
            # FIXME This seems to use identifiers across namespaces. Why not
            # check whether the function is a scoped function first? ~AK
            if insn.expression.function.name == callee_kernel.name:
                caller_kernel = _inline_call_instruction(
                        caller_kernel, callee_kernel, insn)
        elif isinstance(insn, (MultiAssignmentBase, CInstruction,
                _DataObliviousInstruction)):
            pass
        else:
            raise NotImplementedError(
                    "Unknown instruction type %s"
                    % type(insn).__name__)

    return caller_kernel


# FIXME This should take a 'within' parameter to be able to only inline
# *some* calls to a kernel, but not others.
def inline_callable_kernel(program, function_name):
    """
    Returns a copy of *kernel* with the callable kernel addressed by
    (scoped) name *function_name* inlined.
    """
    from loopy.preprocess import infer_arg_descr
    from loopy.program import resolve_callables
    program = resolve_callables(program)
    program = infer_arg_descr(program)
    callables_table = program.callables_table
    new_callables = {}
    callee = program[function_name]

    for func_id, in_knl_callable in callables_table.items():
        if isinstance(in_knl_callable, CallableKernel):
            caller = in_knl_callable.subkernel
            in_knl_callable = in_knl_callable.copy(
                    subkernel=_inline_single_callable_kernel(caller,
                        callee, program.callables_table))
        elif isinstance(in_knl_callable, ScalarCallable):
            pass
        else:
            raise NotImplementedError()

        new_callables[func_id] = in_knl_callable

    return program.copy(callables_table=new_callables)

# }}}


# {{{ tools to match caller to callee args by (guessed) automatic reshaping

# (This is undocumented and not recommended, but it is currently needed
# to support Firedrake.)

class DimChanger(IdentityMapper):
    """
    Mapper to change the dimensions of an argument.

    .. attribute:: callee_arg_dict

        A mapping from the argument name (:class:`str`) to instances of
        :class:`loopy.kernel.array.ArrayBase`.

    .. attribute:: desried_shape

        A mapping from argument name (:class:`str`) to an instance of
        :class:`tuple`.
    """
    def __init__(self, callee_arg_dict, desired_shape):
        self.callee_arg_dict = callee_arg_dict
        self.desired_shape = desired_shape

    def map_subscript(self, expr):
        if expr.aggregate.name not in self.callee_arg_dict:
            return super().map_subscript(expr)
        callee_arg_dim_tags = self.callee_arg_dict[expr.aggregate.name].dim_tags
        flattened_index = sum(dim_tag.stride*idx for dim_tag, idx in
                zip(callee_arg_dim_tags, expr.index_tuple))
        new_indices = []

        from operator import mul
        from functools import reduce
        stride = reduce(mul, self.desired_shape[expr.aggregate.name], 1)

        for length in self.desired_shape[expr.aggregate.name]:
            stride /= length
            ind = flattened_index // int(stride)
            flattened_index -= (int(stride) * ind)
            new_indices.append(simplify_via_aff(ind))

        return expr.aggregate.index(tuple(new_indices))


def _match_caller_callee_argument_dimension_for_single_kernel(
        caller_knl, callee_knl):
    """
    :returns: a copy of *caller_knl* with the instance of
        :class:`loopy.kernel.function_interface.CallableKernel` addressed by
        *callee_function_name* in the *caller_knl* aligned with the argument
        dimensions required by *caller_knl*.
    """
    for insn in caller_knl.instructions:
        if not isinstance(insn, CallInstruction) or (
                insn.expression.function.name !=
                callee_knl.name):
            # Call to a callable kernel can only occur through a
            # CallInstruction.
            continue

        def _shape_1_if_empty(shape):
            assert isinstance(shape, tuple)
            if shape == ():
                return (1, )
            else:
                return shape

        from loopy.kernel.function_interface import (
                ArrayArgDescriptor, get_arg_descriptor_for_expression,
                get_kw_pos_association)
        _, pos_to_kw = get_kw_pos_association(callee_knl)
        arg_id_to_shape = {}
        for arg_id, arg in insn.arg_id_to_val().items():
            arg_id = pos_to_kw[arg_id]

            arg_descr = get_arg_descriptor_for_expression(caller_knl, arg)
            if isinstance(arg_descr, ArrayArgDescriptor):
                arg_id_to_shape[arg_id] = _shape_1_if_empty(arg_descr.shape)
            else:
                arg_id_to_shape[arg_id] = (1, )

        dim_changer = DimChanger(
                callee_knl.arg_dict,
                arg_id_to_shape)

        new_callee_insns = []
        for callee_insn in callee_knl.instructions:
            if isinstance(callee_insn, MultiAssignmentBase):
                new_callee_insns.append(callee_insn.copy(expression=dim_changer(
                    callee_insn.expression),
                    assignee=dim_changer(callee_insn.assignee)))

            elif isinstance(callee_insn, (CInstruction,
                    _DataObliviousInstruction)):
                pass
            else:
                raise NotImplementedError("Unknown instruction %s." %
                        type(insn))

        # subkernel with instructions adjusted according to the new dimensions
        new_callee_knl = callee_knl.copy(instructions=new_callee_insns)

        return new_callee_knl


class _FunctionCalledChecker(CombineMapper):
    def __init__(self, func_name):
        self.func_name = func_name

    def combine(self, values):
        return any(values)

    def map_call(self, expr):
        if expr.function.name == self.func_name:
            return True
        return self.combine(
                tuple(
                    self.rec(child) for child in expr.parameters)
                )

    map_call_with_kwargs = map_call

    def map_constant(self, expr):
        return False

    def map_algebraic_leaf(self, expr):
        return False

    def map_kernel(self, kernel):
        return any(self.rec(insn.expression) for insn in kernel.instructions if
                isinstance(insn, MultiAssignmentBase))


def _match_caller_callee_argument_dimension_(program, callee_function_name):
    """
    Returns a copy of *program* with the instance of
    :class:`loopy.kernel.function_interface.CallableKernel` addressed by
    *callee_function_name* in the *program* aligned with the argument
    dimensions required by *caller_knl*.

    .. note::

        The callee kernel addressed by *callee_function_name*, should be
        called at only one location throughout the program, as multiple
        invocations would demand complex renaming logic which is not
        implemented yet.
    """

    # {{{  sanity checks

    assert isinstance(program, Program)
    assert isinstance(callee_function_name, str)
    assert callee_function_name not in program.entrypoints
    assert callee_function_name in program.callables_table

    # }}}

    is_invoking_callee = _FunctionCalledChecker(
            callee_function_name).map_kernel

    caller_knl,  = [in_knl_callable.subkernel for in_knl_callable in
            program.callables_table.values() if isinstance(in_knl_callable,
                CallableKernel) and
            is_invoking_callee(in_knl_callable.subkernel)]

    from pymbolic.primitives import Call
    assert len([insn for insn in caller_knl.instructions if (isinstance(insn,
        CallInstruction) and isinstance(insn.expression, Call) and
        insn.expression.function.name == callee_function_name)]) == 1
    new_callee_kernel = _match_caller_callee_argument_dimension_for_single_kernel(
            caller_knl, program[callee_function_name])
    return program.with_kernel(new_callee_kernel)

# }}}


def rename_callable(program, old_name, new_name=None, existing_ok=False):
    """
    :arg program: An instance of :class:`loopy.Program`
    :arg old_name: The callable to be renamed
    :arg new_name: New name for the callable to be renamed
    :arg existing_ok: An instance of :class:`bool`
    """
    from loopy.symbolic import (
            RuleAwareSubstitutionMapper,
            SubstitutionRuleMappingContext)
    from pymbolic import var

    assert isinstance(program, Program)
    assert isinstance(old_name, str)

    if (new_name in program.callables_table) and not existing_ok:
        raise LoopyError(f"callables named '{new_name}' already exists")

    if new_name is None:
        namegen = UniqueNameGenerator(program.callables_table.keys())
        new_name = namegen(old_name)

    assert isinstance(new_name, str)

    new_callables_table = {}

    for name, clbl in program.callables_table.items():
        if name == old_name:
            name = new_name

        if isinstance(clbl, CallableKernel):
            knl = clbl.subkernel
            rule_mapping_context = SubstitutionRuleMappingContext(
                    knl.substitutions, knl.get_var_name_generator())
            smap = RuleAwareSubstitutionMapper(rule_mapping_context,
                                               {var(old_name): var(new_name)}.get,
                                               within=lambda *args: True)
            knl = rule_mapping_context.finish_kernel(smap.map_kernel(knl))
            clbl = clbl.copy(subkernel=knl.copy(name=name))
        elif isinstance(clbl, ScalarCallable):
            pass
        else:
            raise NotImplementedError(f"{type(clbl)}")

        new_callables_table[name] = clbl

    new_entrypoints = program.entrypoints.copy()
    if old_name in new_entrypoints:
        new_entrypoints = ((new_entrypoints | frozenset([new_name]))
                           - frozenset([old_name]))

    return program.copy(callables_table=new_callables_table,
                        entrypoints=new_entrypoints)


# vim: foldmethod=marker

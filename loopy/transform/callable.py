from __future__ import division, absolute_import

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

import six

import islpy as isl
from pymbolic.primitives import CallWithKwargs

from loopy.kernel import LoopKernel
from pytools import ImmutableRecord
from loopy.diagnostic import LoopyError
from loopy.kernel.instruction import (CallInstruction, MultiAssignmentBase,
        CInstruction, _DataObliviousInstruction)
from loopy.symbolic import IdentityMapper, SubstitutionMapper, CombineMapper
from loopy.isl_helpers import simplify_via_aff
from loopy.kernel.function_interface import (get_kw_pos_association,
        change_names_of_pymbolic_calls, CallableKernel, ScalarCallable)
from loopy.program import Program, ResolvedFunctionMarker

__doc__ = """
.. currentmodule:: loopy

.. autofunction:: register_function_id_to_in_knl_callable_mapper

.. autofunction:: register_callable_kernel
"""


# {{{ register function lookup

def resolved_callables_from_function_lookup(program,
        func_id_to_kernel_callable_mapper):
    program_callables_info = program.program_callables_info
    program_callables_info = program_callables_info.with_edit_callables_mode()

    callable_knls = dict(
            (func_id, in_knl_callable) for func_id, in_knl_callable in
            program_callables_info.items() if isinstance(in_knl_callable,
                CallableKernel))
    edited_callable_knls = {}

    for func_id, in_knl_callable in callable_knls.items():
        kernel = in_knl_callable.subkernel

        from loopy.symbolic import SubstitutionRuleMappingContext
        rule_mapping_context = SubstitutionRuleMappingContext(
                kernel.substitutions, kernel.get_var_name_generator())

        resolved_function_marker = ResolvedFunctionMarker(
                rule_mapping_context, kernel, program_callables_info,
                [func_id_to_kernel_callable_mapper])

        # scoping fucntions and collecting the scoped functions
        new_subkernel = rule_mapping_context.finish_kernel(
                resolved_function_marker.map_kernel(kernel))
        program_callables_info = resolved_function_marker.program_callables_info

        edited_callable_knls[func_id] = in_knl_callable.copy(
                subkernel=new_subkernel)

    program_callables_info = (
            program_callables_info.with_exit_edit_callables_mode())

    new_resolved_functions = {}

    for func_id, in_knl_callable in program_callables_info.items():
        if func_id in edited_callable_knls:
            new_resolved_functions[func_id] = edited_callable_knls[func_id]
        else:
            new_resolved_functions[func_id] = in_knl_callable

    program_callables_info = program_callables_info.copy(
            resolved_functions=new_resolved_functions)

    return program.copy(program_callables_info=program_callables_info)


def register_function_id_to_in_knl_callable_mapper(program,
        func_id_to_in_knl_callable_mapper):
    """
    Returns a copy of *kernel* with the *function_lookup* registered.

    :arg func_id_to_in_knl_callable_mapper: A function of signature ``(target,
        identifier)`` returning a
        :class:`loopy.kernel.function_interface.InKernelCallable` or *None* if
        the *function_identifier* is not known.
    """

    # adding the function lookup to the set of function lookers in the kernel.
    if func_id_to_in_knl_callable_mapper not in (
            program.func_id_to_in_knl_callable_mappers):
        from loopy.tools import unpickles_equally
        if not unpickles_equally(func_id_to_in_knl_callable_mapper):
            raise LoopyError("function '%s' does not "
                    "compare equally after being upickled "
                    "and would disrupt loopy's caches"
                    % func_id_to_in_knl_callable_mapper)
        new_func_id_mappers = program.func_id_to_in_knl_callable_mappers + (
                [func_id_to_in_knl_callable_mapper])

    program = resolved_callables_from_function_lookup(program,
            func_id_to_in_knl_callable_mapper)

    new_program = program.copy(
            func_id_to_in_knl_callable_mappers=new_func_id_mappers)

    return new_program

# }}}


# {{{ register_callable_kernel

class _RegisterCalleeKernel(ImmutableRecord):
    """
    Helper class to make the function scoper from
    :func:`loopy.transform.register_callable_kernel` picklable. As python
    cannot pickle lexical closures.
    """
    fields = set(['callable_kernel'])

    def __init__(self, callable_kernel):
        self.callable_kernel = callable_kernel

    def __call__(self, target, identifier):
        if identifier == self.callable_kernel.subkernel.name:
            return self.callable_kernel
        return None


def register_callable_kernel(program, callee_kernel):
    """Returns a copy of *caller_kernel*, which would resolve *function_name* in an
    expression as a call to *callee_kernel*.

    :arg caller_kernel: An instance of :class:`loopy.kernel.LoopKernel`.
    :arg function_name: An instance of :class:`str`.
    :arg callee_kernel: An instance of :class:`loopy.kernel.LoopKernel`.
    """

    # {{{ sanity checks

    assert isinstance(program, Program)
    assert isinstance(callee_kernel, LoopKernel)

    # check to make sure that the variables with 'out' direction is equal to
    # the number of assigness in the callee kernel intructions.
    expected_num_assignees = len([arg for arg in callee_kernel.args if
        arg.is_output_only])
    expected_num_parameters = len(callee_kernel.args) - expected_num_assignees
    for in_knl_callable in program.program_callables_info.values():
        if isinstance(in_knl_callable, CallableKernel):
            caller_kernel = in_knl_callable.subkernel
            for insn in caller_kernel.instructions:
                if isinstance(insn, CallInstruction) and (
                        insn.expression.function.name == callee_kernel.name):
                    if isinstance(insn.expression, CallWithKwargs):
                        kw_parameters = insn.expression.kw_parameters
                    else:
                        kw_parameters = {}
                    if len(insn.assignees) != expected_num_assignees:
                        raise LoopyError("The number of arguments with 'out' "
                                "direction " "in callee kernel %s and the number "
                                "of assignees in " "instruction %s do not "
                                "match." % (
                                    callee_kernel.name, insn.id))
                    if len(insn.expression.parameters+tuple(
                            kw_parameters.values())) != expected_num_parameters:
                        raise LoopyError("The number of expected arguments "
                                "for the callee kernel %s and the number of "
                                "parameters in instruction %s do not match."
                                % (callee_kernel.name, insn.id))

                elif isinstance(insn, (MultiAssignmentBase, CInstruction,
                        _DataObliviousInstruction)):
                    pass
                else:
                    raise NotImplementedError("unknown instruction %s" % type(insn))
        elif isinstance(in_knl_callable, ScalarCallable):
            pass
        else:
            raise NotImplementedError("Unknown callable type %s." %
                    type(in_knl_callable).__name__)

    # }}}

    # take the function resolvers from the Program and resolve the functions in
    # the callee kernel
    program_callables_info = (
            program.program_callables_info.with_edit_callables_mode())

    from loopy.symbolic import SubstitutionRuleMappingContext
    rule_mapping_context = SubstitutionRuleMappingContext(
            callee_kernel.substitutions,
            callee_kernel.get_var_name_generator())

    resolved_function_marker = ResolvedFunctionMarker(
            rule_mapping_context, callee_kernel, program_callables_info,
            program.func_id_to_in_knl_callable_mappers)

    callee_kernel = rule_mapping_context.finish_kernel(
            resolved_function_marker.map_kernel(callee_kernel))
    program_callables_info = resolved_function_marker.program_callables_info

    program_callables_info = (
            program_callables_info.with_exit_edit_callables_mode())
    program = program.copy(program_callables_info=program_callables_info)

    # making the target of the child kernel to be same as the target of parent
    # kernel.
    callable_kernel = CallableKernel(subkernel=callee_kernel.copy(
                        target=program.target,
                        is_called_from_host=False))

    # FIXME disabling global barriers for callee kernel (for now)
    from loopy import set_options
    callee_kernel = set_options(callee_kernel, "disable_global_barriers")

    # FIXME: the number of callables is wrong. This is horrible please
    # compensate.

    return register_function_id_to_in_knl_callable_mapper(
            program,
            _RegisterCalleeKernel(callable_kernel))

# }}}


# {{{ callee scoped calls collector (to support inlining)

class CalleeScopedCallsCollector(CombineMapper):
    """
    Collects the scoped functions which are a part of the callee kernel and
    must be transferred to the caller kernel before inlining.

    :returns:
        An :class:`frozenset` of function names that are not scoped in
        the caller kernel.

    .. note::
        :class:`loopy.library.reduction.ArgExtOp` are ignored, as they are
        never scoped in the pipeline.
    """

    def __init__(self, callee_scoped_functions):
        self.callee_scoped_functions = callee_scoped_functions

    def combine(self, values):
        import operator
        from functools import reduce
        return reduce(operator.or_, values, frozenset())

    def map_call(self, expr):
        if expr.function.name in self.callee_scoped_functions:
            return (frozenset([(expr,
                self.callee_scoped_functions[expr.function.name])]) |
                    self.combine((self.rec(child) for child in expr.parameters)))
        else:
            return self.combine((self.rec(child) for child in expr.parameters))

    def map_call_with_kwargs(self, expr):
        if expr.function.name in self.callee_scoped_functions:
            return (frozenset([(expr,
                self.callee_scoped_functions[expr.function.name])]) |
                    self.combine((self.rec(child) for child in expr.parameters
                        + tuple(expr.kw_parameters.values()))))
        else:
            return self.combine((self.rec(child) for child in
                expr.parameters+tuple(expr.kw_parameters.values())))

    def map_constant(self, expr):
        return frozenset()

    map_variable = map_constant
    map_function_symbol = map_constant
    map_tagged_variable = map_constant
    map_type_cast = map_constant

# }}}


# {{{ kernel inliner mapper

class KernelInliner(SubstitutionMapper):
    """Mapper to replace variables (indices, temporaries, arguments) in the
    callee kernel with variables in the caller kernel.

    :arg caller: the caller kernel
    :arg arg_map: dict of argument name to variables in caller
    :arg arg_dict: dict of argument name to arguments in callee
    """

    def __init__(self, subst_func, caller, arg_map, arg_dict):
        super(KernelInliner, self).__init__(subst_func)
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
                    "Argument: {0} in callee kernel: {1} does not have "
                    "constant shape.".format(callee_arg))

            flatten_index = 0
            for i, idx in enumerate(sar.get_begin_subscript().index_tuple):
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
            return super(KernelInliner, self).map_subscript(expr)

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
    for name, temp in six.iteritems(callee_knl.temporary_variables):
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

    if isinstance(instruction.expression, CallWithKwargs):
        from loopy.kernel.function_interface import get_kw_pos_association

        _, pos_to_kw = get_kw_pos_association(callee_knl)
        kw_parameters = instruction.expression.kw_parameters
        for i in range(len(parameters), len(parameters) + len(kw_parameters)):
            parameters = parameters + (kw_parameters[pos_to_kw[i]],)

    assignee_pos = 0
    parameter_pos = 0
    for i, arg in enumerate(callee_knl.args):
        if arg.is_output_only:
            arg_map[arg.name] = assignees[assignee_pos]
            assignee_pos += 1
        else:
            arg_map[arg.name] = parameters[parameter_pos]
            parameter_pos += 1

    # }}}

    # {{{ rewrite instructions

    import pymbolic.primitives as p
    from pymbolic.mapper.substitutor import make_subst_func

    var_map = dict((p.Variable(k), p.Variable(v))
                   for k, v in six.iteritems(iname_map))
    var_map.update(dict((p.Variable(k), p.Variable(v))
                        for k, v in six.iteritems(temp_map)))
    var_map.update(dict((p.Variable(k), p.Variable(v.subscript.aggregate.name))
                        for k, v in six.iteritems(arg_map)))
    subst_mapper = KernelInliner(
        make_subst_func(var_map), kernel, arg_map, callee_knl.arg_dict)

    insn_id = {}
    for insn in callee_knl.instructions:
        insn_id[insn.id] = ing(callee_label+insn.id)

    # {{{ root and leave instructions in callee kernel

    dep_map = callee_knl.recursive_insn_dep_map()
    # roots depend on nothing
    heads = set(insn for insn, deps in six.iteritems(dep_map) if not deps)
    # leaves have nothing that depends on them
    tails = set(dep_map.keys())
    for insn, deps in six.iteritems(dep_map):
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
            depends_on = depends_on | set([noop_start.id])
        insn = insn.copy(
            id=insn_id[insn.id],
            within_inames=within_inames,
            # TODO: probaby need to keep priority in callee kernel
            priority=instruction.priority,
            depends_on=depends_on
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

def _inline_single_callable_kernel(caller_kernel, function_name,
        program_callables_info):
    old_insns = caller_kernel.instructions
    for insn in old_insns:
        if isinstance(insn, CallInstruction):
            # FIXME This seems to use identifiers across namespaces. Why not
            # check whether the function is a scoped function first? ~AK
            if insn.expression.function.name in program_callables_info:
                history_of_identifier = program_callables_info.history[
                        insn.expression.function.name]

                if function_name in history_of_identifier:
                    in_knl_callable = program_callables_info[
                            insn.expression.function.name]
                    assert isinstance(in_knl_callable, CallableKernel)
                    caller_kernel = _inline_call_instruction(
                            caller_kernel, in_knl_callable.subkernel, insn)
                    program_callables_info = (
                            program_callables_info.with_deleted_callable(
                                insn.expression.function.name,
                                program_callables_info.num_times_callables_called[
                                    caller_kernel.name]))
        elif isinstance(insn, (MultiAssignmentBase, CInstruction,
                _DataObliviousInstruction)):
            pass
        else:
            raise NotImplementedError(
                    "Unknown instruction type %s"
                    % type(insn).__name__)

    return caller_kernel, program_callables_info


# FIXME This should take a 'within' parameter to be able to only inline
# *some* calls to a kernel, but not others.
def inline_callable_kernel(program, function_name):
    """
    Returns a copy of *kernel* with the callable kernel addressed by
    (scoped) name *function_name* inlined.
    """
    from loopy.preprocess import infer_arg_descr
    program = infer_arg_descr(program)
    program_callables_info = program.program_callables_info
    old_program_callables_info = program_callables_info.copy()

    edited_callable_kernels = {}

    for func_id, in_knl_callable in old_program_callables_info.items():
        if function_name not in old_program_callables_info.history[func_id] and (
                isinstance(in_knl_callable, CallableKernel)):
            caller_kernel = in_knl_callable.subkernel
            caller_kernel, program_callables_info = (
                    _inline_single_callable_kernel(caller_kernel,
                        function_name,
                        program_callables_info))
            edited_callable_kernels[func_id] = in_knl_callable.copy(
                    subkernel=caller_kernel)

    new_resolved_functions = {}
    for func_id, in_knl_callable in program_callables_info.items():
        if func_id in edited_callable_kernels:
            new_resolved_functions[func_id] = edited_callable_kernels[func_id]
        else:
            new_resolved_functions[func_id] = in_knl_callable

    program_callables_info = program_callables_info.copy(
            resolved_functions=new_resolved_functions)

    return program.copy(program_callables_info=program_callables_info)

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
        caller_knl, callee_function_name):
    """
    Returns a copy of *caller_knl* with the instance of
    :class:`loopy.kernel.function_interface.CallableKernel` addressed by
    *callee_function_name* in the *caller_knl* aligned with the argument
    dimesnsions required by *caller_knl*.
    """
    pymbolic_calls_to_new_callables = {}
    for insn in caller_knl.instructions:
        if not isinstance(insn, CallInstruction) or (
                insn.expression.function.name not in
                caller_knl.scoped_functions):
            # Call to a callable kernel can only occur through a
            # CallInstruction.
            continue

        in_knl_callable = caller_knl.scoped_functions[
                insn.expression.function.name]

        if in_knl_callable.subkernel.name != callee_function_name:
            # Not the callable we're looking for.
            continue

        # getting the caller->callee arg association

        parameters = insn.expression.parameters[:]
        kw_parameters = {}
        if isinstance(insn.expression, CallWithKwargs):
            kw_parameters = insn.expression.kw_parameters

        assignees = insn.assignees

        parameter_shapes = [par.get_array_arg_descriptor(caller_knl).shape
                for par in parameters]
        kw_to_pos, pos_to_kw = get_kw_pos_association(in_knl_callable.subkernel)
        for i in range(len(parameters), len(parameters)+len(kw_parameters)):
            parameter_shapes.append(kw_parameters[pos_to_kw[i]]
                    .get_array_arg_descriptor(caller_knl).shape)

        # inserting the assigness at the required positions.
        assignee_write_count = -1
        for i, arg in enumerate(in_knl_callable.subkernel.args):
            if arg.is_output_only:
                assignee = assignees[-assignee_write_count-1]
                parameter_shapes.insert(i, assignee
                        .get_array_arg_descriptor(caller_knl).shape)
                assignee_write_count -= 1

        callee_arg_to_desired_dim_tag = dict(zip([arg.name for arg in
            in_knl_callable.subkernel.args], parameter_shapes))
        dim_changer = DimChanger(in_knl_callable.subkernel.arg_dict,
                callee_arg_to_desired_dim_tag)
        new_callee_insns = []
        for callee_insn in in_knl_callable.subkernel.instructions:
            if isinstance(callee_insn, MultiAssignmentBase):
                new_callee_insns.append(callee_insn.copy(expression=dim_changer(
                    callee_insn.expression),
                    assignee=dim_changer(callee_insn.assignee)))
            elif isinstance(callee_insn, (CInstruction,
                    _DataObliviousInstruction)):
                pass
            else:
                raise NotImplementedError("Unknwon instruction %s." %
                        type(insn))

        # subkernel with instructions adjusted according to the new dimensions.
        new_subkernel = in_knl_callable.subkernel.copy(instructions=new_callee_insns)

        new_in_knl_callable = in_knl_callable.copy(subkernel=new_subkernel)

        pymbolic_calls_to_new_callables[insn.expression] = new_in_knl_callable

    if not pymbolic_calls_to_new_callables:
        # complain if no matching function found.
        raise LoopyError("No CallableKernel with the name %s found in %s." % (
            callee_function_name, caller_knl.name))

    return change_names_of_pymbolic_calls(caller_knl,
            pymbolic_calls_to_new_callables)


def _match_caller_callee_argument_dimension_(program, *args, **kwargs):
    assert isinstance(program, Program)

    new_resolved_functions = {}
    for func_id, in_knl_callable in program.program_callables_info.items():
        if isinstance(in_knl_callable, CallableKernel):
            new_subkernel = (
                    _match_caller_callee_argument_dimension_for_single_kernel(
                        in_knl_callable.subkernel, program.program_callables_info,
                        *args, **kwargs))
            in_knl_callable = in_knl_callable.copy(
                    subkernel=new_subkernel)

        elif isinstance(in_knl_callable, ScalarCallable):
            pass
        else:
            raise NotImplementedError("Unknown type of callable %s." % (
                type(in_knl_callable).__name__))

        new_resolved_functions[func_id] = in_knl_callable

    new_program_callables_info = program.program_callables_info.copy(
            resolved_functions=new_resolved_functions)
    return program.copy(program_callables_info=new_program_callables_info)

# }}}


# vim: foldmethod=marker

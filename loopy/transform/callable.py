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
from loopy.symbolic import (
            RuleAwareSubstitutionMapper,
            SubstitutionRuleMappingContext, CombineMapper, IdentityMapper)
from loopy.isl_helpers import simplify_via_aff
from loopy.kernel.function_interface import (
        CallableKernel, ScalarCallable)
from loopy.program import Program

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
        raise LoopyError("Redefining function identifier not allowed. Set the"
                " option 'redefining_not_ok=False' to bypass this error.")

    new_callables = translation_unit.callables_table.set(function_identifier,
            callable_)

    return translation_unit.copy(
            callables_table=new_callables)


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

class KernelInliner(RuleAwareSubstitutionMapper):
    def __init__(self, rule_mapping_context, subst_func, caller_knl,
            callee_knl, callee_arg_to_call_param):
        super().__init__(rule_mapping_context, subst_func, lambda *args: True)
        self.caller_knl = caller_knl
        self.callee_knl = callee_knl
        self.callee_arg_to_call_param = callee_arg_to_call_param

    def map_subscript(self, expr, expn_state):
        if expr.aggregate.name in self.callee_knl.arg_dict:
            from loopy.symbolic import get_start_subscript_from_sar
            from loopy.isl_helpers import simplify_via_aff
            from pymbolic.primitives import Subscript, Variable

            sar = self.callee_arg_to_call_param[expr.aggregate.name]  # SubArrayRef

            callee_arg = self.callee_knl.arg_dict[expr.aggregate.name]
            if sar.subscript.aggregate.name in self.caller_knl.arg_dict:
                caller_arg = self.caller_knl.arg_dict[sar.subscript.aggregate.name]
            else:
                caller_arg = self.caller_knl.temporary_variables[
                        sar.subscript.aggregate.name]

            # map inner inames to outer inames.
            outer_indices = self.map_tuple(expr.index_tuple, expn_state)

            flatten_index = 0
            for i, idx in enumerate(get_start_subscript_from_sar(sar,
                    self.caller_knl).index_tuple):
                flatten_index += idx*caller_arg.dim_tags[i].stride

            flatten_index += sum(
                idx * tag.stride
                for idx, tag in zip(outer_indices, callee_arg.dim_tags))

            flatten_index = simplify_via_aff(flatten_index)

            new_indices = []
            for dim_tag in caller_arg.dim_tags:
                ind = flatten_index // dim_tag.stride
                flatten_index -= (dim_tag.stride * ind)
                new_indices.append(ind)

            new_indices = tuple(simplify_via_aff(i) for i in new_indices)

            return Subscript(Variable(sar.subscript.aggregate.name), new_indices)
        else:
            assert expr.aggregate.name in self.callee_knl.temporary_variables
            return super().map_subscript(expr, expn_state)

# }}}


# {{{ inlining of a single call instruction

def substitute_into_domain(domain, param_name, expr, allowed_param_dims):
    """
    :arg allowed_deps: A :class:`list` of :class:`str` that are
    """
    import pymbolic.primitives as prim
    from loopy.symbolic import get_dependencies, isl_set_from_expr
    if param_name not in domain.get_var_dict():
        # param_name not in domain => domain will be unchanged
        return domain

    # {{{ rename 'param_name' to avoid namespace pollution with allowed_param_dims

    dt, pos = domain.get_var_dict()[param_name]
    domain = domain.set_dim_name(dt, pos, UniqueNameGenerator(
        set(allowed_param_dims))(param_name))

    # }}}

    for dep in get_dependencies(expr):
        if dep in allowed_param_dims:
            domain = domain.add_dims(isl.dim_type.param, 1)
            domain = domain.set_dim_name(
                    isl.dim_type.param,
                    domain.dim(isl.dim_type.param)-1,
                    dep)
        else:
            raise ValueError("Augmenting caller's domain "
                    f"with '{dep}' is not allowed.")

    set_ = isl_set_from_expr(domain.space,
            prim.Comparison(prim.Variable(param_name),
                            "==",
                            expr))

    bset, = set_.get_basic_sets()
    domain = domain & bset

    return domain.project_out(dt, pos, 1)


def rename_iname(domain, old_iname, new_iname):
    if old_iname not in domain.get_var_dict():
        return domain

    dt, pos = domain.get_var_dict()[old_iname]
    return domain.set_dim_name(dt, pos, new_iname)


def get_valid_domain_param_names(knl):
    from loopy.kernel.data import ValueArg
    return ([arg.name for arg in knl.args if isinstance(arg, ValueArg)]
            + [tv.name
               for tv in knl.temporary_variables.values()
               if tv.shape == ()]
            + list(knl.all_inames())
            )


def _inline_call_instruction(caller_knl, callee_knl, call_insn):
    """
    Returns a copy of *caller_knl* with the *call_insn* in the *kernel*
    replaced by inlining *callee_knl* into it within it.
    """
    import pymbolic.primitives as prim
    from pymbolic.mapper.substitutor import make_subst_func
    from loopy.kernel.data import ValueArg

    # {{{ sanity checks

    assert call_insn.expression.function.name == callee_knl.name

    # }}}

    callee_label = callee_knl.name[:4] + "_"
    vng = caller_knl.get_var_name_generator()
    ing = caller_knl.get_instruction_id_generator()

    # {{{ construct callee->caller name mappings

    # name_map: Mapping[str, str]
    # A mapping from variable names in the callee kernel's namespace to
    # the ones they would be referred by in the caller's namespace post inlining.
    name_map = {}

    # only consider temporary variables and inames, arguments would be mapping
    # according to the invocation in call_insn.
    for name in (callee_knl.all_inames()
                 | set(callee_knl.temporary_variables.keys())):
        new_name = vng(callee_label+name)
        name_map[name] = new_name

    # }}}

    # {{{ iname_to_tags

    # new_iname_to_tags: caller's iname_to_tags post inlining
    new_iname_to_tags = caller_knl.iname_to_tags

    for old_name, tags in callee_knl.iname_to_tags.items():
        new_iname_to_tags[name_map[old_name]] = tags

    # }}}

    # {{{ register callee's temps as caller's

    # new_temps: caller's temps post inlining
    new_temps = caller_knl.temporary_variables.copy()

    for name, tv in callee_knl.temporary_variables.items():
        new_temps[name_map[name]] = tv.copy(name=name_map[name])

    # }}}

    # {{{ get callee args -> parameters passed to the call

    arg_map = {}  # callee arg name -> caller symbols (e.g. SubArrayRef)

    assignees = call_insn.assignees  # writes
    parameters = call_insn.expression.parameters  # reads

    # add keyword parameters
    from pymbolic.primitives import CallWithKwargs

    from loopy.kernel.function_interface import get_kw_pos_association
    kw_to_pos, pos_to_kw = get_kw_pos_association(callee_knl)
    if isinstance(call_insn.expression, CallWithKwargs):
        kw_parameters = call_insn.expression.kw_parameters
    else:
        kw_parameters = {}

    for kw, par in kw_parameters.items():
        arg_map[kw] = par

    for i, par in enumerate(parameters):
        arg_map[pos_to_kw[i]] = par

    for i, assignee in enumerate(assignees):
        arg_map[pos_to_kw[-i-1]] = assignee

    # }}}

    # {{{ domains/assumptions

    new_domains = callee_knl.domains.copy()
    for old_iname in callee_knl.all_inames():
        new_domains = [rename_iname(dom, old_iname, name_map[old_iname])
                       for dom in new_domains]

    new_assumptions = callee_knl.assumptions

    for callee_arg_name, param_expr in arg_map.items():
        if isinstance(callee_knl.arg_dict[callee_arg_name],
                      ValueArg):
            new_domains = [
                    substitute_into_domain(
                        dom,
                        callee_arg_name,
                        param_expr, get_valid_domain_param_names(caller_knl))
                    for dom in new_domains]

            new_assumptions = substitute_into_domain(
                        new_assumptions,
                        callee_arg_name,
                        param_expr, get_valid_domain_param_names(caller_knl))

    # }}}

    # {{{ map callee's expressions to get expressions after inlining

    rule_mapping_context = SubstitutionRuleMappingContext(
            callee_knl.substitutions, vng)
    smap = KernelInliner(rule_mapping_context,
            make_subst_func({old_name: prim.Variable(new_name)
                             for old_name, new_name in name_map.items()}),
            caller_knl, callee_knl, arg_map)

    callee_knl = rule_mapping_context.finish_kernel(smap.map_kernel(
        callee_knl))

    # }}}

    # {{{ generate new ids for instructions

    insn_id_map = {}
    for insn in callee_knl.instructions:
        insn_id_map[insn.id] = ing(callee_label+insn.id)

    # }}}

    # {{{ use NoOp to mark the start and end of callee kernel

    from loopy.kernel.instruction import NoOpInstruction

    noop_start = NoOpInstruction(
        id=ing(callee_label+"_start"),
        within_inames=call_insn.within_inames,
        depends_on=call_insn.depends_on
    )
    noop_end = NoOpInstruction(
        id=call_insn.id,
        within_inames=call_insn.within_inames,
        depends_on=frozenset(insn_id_map.values())
    )

    # }}}

    # {{{ map callee's instruction ids

    inlined_insns = [noop_start]

    for insn in callee_knl.instructions:
        new_within_inames = (frozenset(name_map[iname]
                             for iname in insn.within_inames)
                | call_insn.within_inames)
        new_depends_on = (frozenset(insn_id_map[dep] for dep in insn.depends_on)
                      | {noop_start.id})
        new_no_sync_with = frozenset((insn_id_map[id], scope)
                                 for id, scope in insn.no_sync_with)
        new_id = insn_id_map[insn.id]

        if isinstance(insn, Assignment):
            new_atomicity = tuple(
                    type(atomicity)(name_map[atomicity.var_name])
                    for atomicity in insn.atomicity)
            insn = insn.copy(
                id=insn_id_map[insn.id],
                within_inames=new_within_inames,
                depends_on=new_depends_on,
                tags=insn.tags | call_insn.tags,
                atomicity=new_atomicity,
                no_sync_with=new_no_sync_with
            )
        else:
            insn = insn.copy(
                id=new_id,
                within_inames=new_within_inames,
                depends_on=new_depends_on,
                tags=insn.tags | call_insn.tags,
                no_sync_with=new_no_sync_with
            )
        inlined_insns.append(insn)

    inlined_insns.append(noop_end)

    # }}}

    # {{{ swap out call_insn with inlined_instructions

    idx = caller_knl.instructions.index(call_insn)
    new_insns = (caller_knl.instructions[:idx]
                 + inlined_insns
                 + caller_knl.instructions[idx+1:])

    # }}}

    old_assumptions, new_assumptions = isl.align_two(
            caller_knl.assumptions, new_assumptions)

    return caller_knl.copy(instructions=new_insns,
            temporary_variables=new_temps,
            domains=caller_knl.domains+new_domains,
            assumptions=old_assumptions.params() & new_assumptions.params(),
            iname_to_tags=new_iname_to_tags)

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

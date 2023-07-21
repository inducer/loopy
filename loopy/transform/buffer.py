__copyright__ = "Copyright (C) 2012-2015 Andreas Kloeckner"

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

from immutables import Map
from loopy.transform.array_buffer_map import (ArrayToBufferMap, NoOpArrayToBufferMap,
        AccessDescriptor)
from loopy.symbolic import (get_dependencies,
        RuleAwareIdentityMapper, SubstitutionRuleMappingContext,
        SubstitutionMapper)
from pymbolic.mapper.substitutor import make_subst_func
from loopy.tools import memoize_on_disk
from loopy.diagnostic import LoopyError
from loopy.kernel import LoopKernel
from loopy.translation_unit import TranslationUnit
from loopy.kernel.function_interface import CallableKernel, ScalarCallable

from pymbolic import var

import logging
logger = logging.getLogger(__name__)


# {{{ replace array access

class ArrayAccessReplacer(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context,
            var_name, within, array_base_map, buf_var):
        super().__init__(rule_mapping_context)

        self.within = within

        self.array_base_map = array_base_map

        self.var_name = var_name
        self.modified_insn_ids = set()

        self.buf_var = buf_var

    def map_variable(self, expr, expn_state):
        result = None
        if expr.name == self.var_name and self.within(
                expn_state.kernel,
                expn_state.instruction,
                expn_state.stack):
            result = self.map_array_access((), expn_state)

        if result is None:
            return super().map_variable(expr, expn_state)
        else:
            self.modified_insn_ids.add(expn_state.insn_id)
            return result

    def map_subscript(self, expr, expn_state):
        result = None
        if expr.aggregate.name == self.var_name and self.within(
                expn_state.kernel,
                expn_state.instruction,
                expn_state.stack):
            result = self.map_array_access(expr.index_tuple, expn_state)

        if result is None:
            return super().map_subscript(expr, expn_state)
        else:
            self.modified_insn_ids.add(expn_state.insn_id)
            return result

    def map_array_access(self, index, expn_state):
        accdesc = AccessDescriptor(
            identifier=None,
            storage_axis_exprs=index)

        if not self.array_base_map.is_access_descriptor_in_footprint(accdesc):
            return None

        abm = self.array_base_map

        index = expn_state.apply_arg_context(index)

        assert len(index) == len(abm.non1_storage_axis_flags)

        access_subscript = []
        for i in range(len(index)):
            if not abm.non1_storage_axis_flags[i]:
                continue

            ax_index = index[i]
            from loopy.symbolic import simplify_via_aff
            ax_index = simplify_via_aff(
                    ax_index - abm.storage_base_indices[i])

            access_subscript.append(ax_index)

        result = self.buf_var
        if access_subscript:
            result = result.index(tuple(access_subscript))

        # Can't possibly be nested, but recurse anyway to
        # make sure substitution rules referenced below here
        # do not get thrown away.
        self.rec(result, expn_state.copy(arg_context={}))

        return result

# }}}


def buffer_array_for_single_kernel(kernel, callables_table, var_name,
        buffer_inames, init_expression=None, store_expression=None,
        within=None, default_tag="l.auto", temporary_scope=None,
        fetch_bounding_box=False):
    """Replace accesses to *var_name* with ones to a temporary, which is
    created and acts as a buffer. To perform this transformation, the access
    footprint to *var_name* is determined and a temporary of a suitable
    :class:`loopy.AddressSpace` and shape is created.

    By default, the value of the buffered cells in *var_name* are read prior to
    any (read/write) use, and the modified values are written out after use has
    concluded, but for special use cases (e.g. additive accumulation), the
    behavior can be modified using *init_expression* and *store_expression*.

    :arg buffer_inames: The inames across which the buffer should be usable--i.e.
        all possible values of these inames will be covered by the buffer footprint.
        A tuple of inames or a comma-separated string.
    :arg init_expression: Either *None* (indicating the prior value of the buffered
        array should be read) or an expression optionally involving the
        variable 'base' (which references the associated location in the array
        being buffered).
    :arg store_expression: Either *None*, *False*, or an expression involving
        variables 'base' and 'buffer' (without array indices).
        (*None* indicates that a default storage instruction should be used,
        *False* indicates that no storing of the temporary should occur
        at all.)
    :arg within: If not None, limit the action of the transformation to
        matching contexts.  See :func:`loopy.match.parse_stack_match`
        for syntax.
    :arg temporary_scope: If given, override the choice of
        :class:`AddressSpace` for the created temporary.
    :arg default_tag: The default :ref:`iname-tags` to be assigned to the
        inames used for fetching and storing
    :arg fetch_bounding_box: If the access footprint is non-convex
        (resulting in an error), setting this argument to *True* will force a
        rectangular (and hence convex) superset of the footprint to be
        fetched.
    """

    if isinstance(kernel, TranslationUnit):
        kernel_names = [i for i, clbl in
                kernel.callables_table.items() if isinstance(clbl,
                    CallableKernel)]
        if len(kernel_names) != 1:
            raise LoopyError()

        return kernel.with_kernel(buffer_array(kernel[kernel_names[0]],
            var_name, buffer_inames, init_expression, store_expression, within,
            default_tag, temporary_scope,
            fetch_bounding_box, kernel.callables_table))

    assert isinstance(kernel, LoopKernel)

    # {{{ process arguments

    if isinstance(init_expression, str):
        from loopy.symbolic import parse
        init_expression = parse(init_expression)

    if isinstance(store_expression, str):
        from loopy.symbolic import parse
        store_expression = parse(store_expression)

    if isinstance(buffer_inames, str):
        buffer_inames = [s.strip()
                for s in buffer_inames.split(",") if s.strip()]

    for iname in buffer_inames:
        if iname not in kernel.all_inames():
            raise RuntimeError("sweep iname '%s' is not a known iname"
                    % iname)

    buffer_inames = list(buffer_inames)
    buffer_inames_set = frozenset(buffer_inames)

    from loopy.match import parse_stack_match
    within = parse_stack_match(within)

    if var_name in kernel.arg_dict:
        var_descr = kernel.arg_dict[var_name]
    elif var_name in kernel.temporary_variables:
        var_descr = kernel.temporary_variables[var_name]
    else:
        raise ValueError("variable '%s' not found" % var_name)

    from loopy.kernel.data import ArrayBase
    if isinstance(var_descr, ArrayBase):
        var_shape = var_descr.shape
    else:
        var_shape = ()

    if temporary_scope is None:
        import loopy as lp
        temporary_scope = lp.auto

    # }}}

    var_name_gen = kernel.get_var_name_generator()
    within_inames = set()

    access_descriptors = []
    for insn in kernel.instructions:
        if not within(kernel, insn, ()):
            continue

        from pymbolic.primitives import Variable, Subscript
        from loopy.symbolic import LinearSubscript

        for assignee in insn.assignees:
            if isinstance(assignee, Variable):
                assignee_name = assignee.name
                index = ()

            elif isinstance(assignee, Subscript):
                assignee_name = assignee.aggregate.name
                index = assignee.index_tuple

            elif isinstance(assignee, LinearSubscript):
                if assignee.aggregate.name == var_name:
                    raise LoopyError("buffer_array may not be applied in the "
                            "presence of linear write indexing into '%s'" % var_name)

            else:
                raise LoopyError("invalid lvalue '%s'" % assignee)

            if assignee_name == var_name:
                within_inames.update(
                        (get_dependencies(index) & kernel.all_inames())
                        - buffer_inames_set)
                access_descriptors.append(
                        AccessDescriptor(
                            identifier=insn.id,
                            storage_axis_exprs=index))

    # {{{ find fetch/store inames

    init_inames = []
    store_inames = []
    new_iname_to_tag = {}

    for i in range(len(var_shape)):
        dim_name = str(i)
        if isinstance(var_descr, ArrayBase) and var_descr.dim_names is not None:
            dim_name = var_descr.dim_names[i]

        init_iname = var_name_gen(f"{var_name}_init_{dim_name}")
        store_iname = var_name_gen(f"{var_name}_store_{dim_name}")

        new_iname_to_tag[init_iname] = default_tag
        new_iname_to_tag[store_iname] = default_tag

        init_inames.append(init_iname)
        store_inames.append(store_iname)

    # }}}

    # {{{ modify loop domain

    non1_init_inames = []
    non1_store_inames = []

    if var_shape:
        # {{{ find domain to be changed

        from loopy.kernel.tools import DomainChanger
        domch = DomainChanger(kernel, buffer_inames_set | within_inames)

        if domch.leaf_domain_index is not None:
            # If the sweep inames are at home in parent domains, then we'll add
            # fetches with loops over copies of these parent inames that will end
            # up being scheduled *within* loops over these parents.

            for iname in buffer_inames_set:
                if kernel.get_home_domain_index(iname) != domch.leaf_domain_index:
                    raise RuntimeError("buffer iname '%s' is not 'at home' in the "
                            "sweep's leaf domain" % iname)

        # }}}

        abm = ArrayToBufferMap(kernel, domch.domain, buffer_inames,
                access_descriptors, len(var_shape))

        for i in range(len(var_shape)):
            if abm.non1_storage_axis_flags[i]:
                non1_init_inames.append(init_inames[i])
                non1_store_inames.append(store_inames[i])
            else:
                del new_iname_to_tag[init_inames[i]]
                del new_iname_to_tag[store_inames[i]]

        new_domain = domch.domain
        new_domain = abm.augment_domain_with_sweep(
                    new_domain, non1_init_inames,
                    boxify_sweep=fetch_bounding_box)
        new_domain = abm.augment_domain_with_sweep(
                    new_domain, non1_store_inames,
                    boxify_sweep=fetch_bounding_box)
        new_kernel_domains = domch.get_domains_with(new_domain)
        del new_domain

    else:
        # leave kernel domains unchanged
        new_kernel_domains = kernel.domains

        abm = NoOpArrayToBufferMap()

    # }}}

    # {{{ set up temp variable

    import loopy as lp

    buf_var_name = var_name_gen(based_on=var_name+"_buf")

    new_temporary_variables = kernel.temporary_variables.copy()
    temp_var = lp.TemporaryVariable(
            name=buf_var_name,
            dtype=var_descr.dtype,
            base_indices=(0,)*len(abm.non1_storage_shape),
            shape=tuple(abm.non1_storage_shape),
            address_space=temporary_scope)

    new_temporary_variables[buf_var_name] = temp_var

    # }}}

    new_insns = []

    buf_var = var(buf_var_name)

    # {{{ generate init instruction

    buf_var_init = buf_var
    if non1_init_inames:
        buf_var_init = buf_var_init.index(
                tuple(var(iname) for iname in non1_init_inames))

    init_base = var(var_name)

    init_subscript = []
    init_iname_idx = 0
    if var_shape:
        for i in range(len(var_shape)):
            ax_subscript = abm.storage_base_indices[i]
            if abm.non1_storage_axis_flags[i]:
                ax_subscript += var(non1_init_inames[init_iname_idx])
                init_iname_idx += 1
            init_subscript.append(ax_subscript)

    if init_subscript:
        init_base = init_base.index(tuple(init_subscript))

    if init_expression is None:
        init_expression = init_base
    else:
        init_expression = init_expression
        init_expression = SubstitutionMapper(
                make_subst_func({
                    "base": init_base,
                    }))(init_expression)

    init_insn_id = kernel.make_unique_instruction_id(based_on="init_"+var_name)
    from loopy.kernel.data import Assignment
    init_instruction = Assignment(id=init_insn_id,
                assignee=buf_var_init,
                expression=init_expression,
                within_inames=(
                    frozenset(within_inames)
                    | frozenset(non1_init_inames)),
                depends_on=frozenset(),
                depends_on_is_final=True)

    # }}}

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    aar = ArrayAccessReplacer(rule_mapping_context, var_name,
            within, abm, buf_var)
    kernel = rule_mapping_context.finish_kernel(aar.map_kernel(kernel))

    did_write = False
    for insn_id in aar.modified_insn_ids:
        insn = kernel.id_to_insn[insn_id]
        if buf_var_name in insn.assignee_var_names():
            did_write = True

    # {{{ add init_insn_id to depends_on

    new_insns = []

    def none_to_empty_set(s):
        if s is None:
            return frozenset()
        else:
            return s

    for insn in kernel.instructions:
        if insn.id in aar.modified_insn_ids:
            new_insns.append(
                    insn.copy(
                        depends_on=(
                            none_to_empty_set(insn.depends_on)
                            | frozenset([init_insn_id]))))
        else:
            new_insns.append(insn)

    # }}}

    # {{{ generate store instruction

    buf_var_store = buf_var
    if non1_store_inames:
        buf_var_store = buf_var_store.index(
                tuple(var(iname) for iname in non1_store_inames))

    store_subscript = []
    store_iname_idx = 0
    if var_shape:
        for i in range(len(var_shape)):
            ax_subscript = abm.storage_base_indices[i]
            if abm.non1_storage_axis_flags[i]:
                ax_subscript += var(non1_store_inames[store_iname_idx])
                store_iname_idx += 1
            store_subscript.append(ax_subscript)

    store_target = var(var_name)
    if store_subscript:
        store_target = store_target.index(tuple(store_subscript))

    if store_expression is None:
        store_expression = buf_var_store
    else:
        store_expression = SubstitutionMapper(
                make_subst_func({
                    "base": store_target,
                    "buffer": buf_var_store,
                    }))(store_expression)

    if store_expression is not False:
        from loopy.kernel.data import Assignment
        store_instruction = Assignment(
                    id=kernel.make_unique_instruction_id(based_on="store_"+var_name),
                    depends_on=frozenset(aar.modified_insn_ids),
                    no_sync_with=frozenset([(init_insn_id, "any")]),
                    assignee=store_target,
                    expression=store_expression,
                    within_inames=(
                        frozenset(within_inames)
                        | frozenset(non1_store_inames)))
    else:
        did_write = False

    # }}}

    new_insns.append(init_instruction)
    if did_write:
        # new_insns_with_redirected_deps: if an insn depends on a modified
        # insn, then it should also depend on the store insn.
        new_insns_with_redirected_deps = [
            insn.copy(depends_on=(insn.depends_on | {store_instruction.id}))
            if insn.depends_on & aar.modified_insn_ids
            else insn
            for insn in new_insns
        ] + [store_instruction]
    else:
        for iname in store_inames:
            del new_iname_to_tag[iname]

        new_insns_with_redirected_deps = new_insns

    kernel = kernel.copy(
            domains=new_kernel_domains,
            instructions=new_insns_with_redirected_deps,
            temporary_variables=new_temporary_variables)

    from loopy import tag_inames
    kernel = tag_inames(kernel, new_iname_to_tag)

    from loopy.kernel.tools import assign_automatic_axes
    kernel = assign_automatic_axes(kernel, callables_table)

    return kernel


@memoize_on_disk
def buffer_array(program, *args, **kwargs):
    assert isinstance(program, TranslationUnit)

    new_callables = {}

    for func_id, clbl in program.callables_table.items():
        if isinstance(clbl, CallableKernel):
            clbl = clbl.copy(
                    subkernel=buffer_array_for_single_kernel(clbl.subkernel,
                        program.callables_table, *args, **kwargs))
        elif isinstance(clbl, ScalarCallable):
            pass
        else:
            raise NotImplementedError()

        new_callables[func_id] = clbl

    return program.copy(callables_table=Map(new_callables))


# vim: foldmethod=marker

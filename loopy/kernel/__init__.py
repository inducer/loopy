"""Kernel object."""

from __future__ import division

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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


import numpy as np
from pytools import Record, memoize_method
import islpy as isl
from islpy import dim_type
import re

from pytools import UniqueNameGenerator, generate_unique_possibilities

from loopy.library.function import (
        default_function_mangler,
        opencl_function_mangler,
        single_arg_function_mangler)

from loopy.library.symbol import opencl_symbol_mangler
from loopy.library.preamble import default_preamble_generator

from loopy.diagnostic import CannotBranchDomainTree


# {{{ unique var names

def _is_var_name_conflicting_with_longer(name_a, name_b):
    # Array dimensions implemented as separate arrays generate
    # names by appending '_s<NUMBER>'. Make sure that no
    # conflicts can arise from these names.

    # Only deal with the case of b longer than a.
    if not name_b.startswith(name_a):
        return False

    return re.match("^%s_s[0-9]+" % re.escape(name_b), name_a) is not None


def _is_var_name_conflicting(name_a, name_b):
    if name_a == name_b:
        return True

    return (
            _is_var_name_conflicting_with_longer(name_a, name_b)
            or _is_var_name_conflicting_with_longer(name_b, name_a))


class _UniqueVarNameGenerator(UniqueNameGenerator):
    def is_name_conflicting(self, name):
        from pytools import any
        return any(
                _is_var_name_conflicting(name, other_name)
                for other_name in self.existing_names)

# }}}


# {{{ loop kernel object

class LoopKernel(Record):
    """These correspond more or less directly to arguments of
    :func:`loopy.make_kernel`.

    .. attribute:: device

        :class:`pyopencl.Device`

    .. attribute:: domains

        a list of :class:`islpy.BasicSet` instances

    .. attribute:: instructions
    .. attribute:: args
    .. attribute:: schedule

        *None* or a list of :class:`loopy.schedule.ScheduleItem`

    .. attribute:: name
    .. attribute:: preambles
    .. attribute:: preamble_generators
    .. attribute:: assumptions
    .. attribute:: local_sizes
    .. attribute:: temporary_variables
    .. attribute:: iname_to_tag
    .. attribute:: function_manglers
    .. attribute:: symbol_manglers

    .. attribute:: substitutions

        a mapping from substitution names to
        :class:`SubstitutionRule` objects

    .. attribute:: iname_slab_increments

        a dictionary mapping inames to (lower_incr,
        upper_incr) tuples that will be separated out in the execution to generate
        'bulk' slabs with fewer conditionals.

    .. attribute:: loop_priority

        A list of inames. The earlier in the list the iname occurs, the earlier
        it will be scheduled. (This applies to inames with non-parallel
        implementation tags.)

    .. attribute:: silenced_warnings

    .. attribute:: applied_iname_rewrites

        A list of past substitution dictionaries that
        were applied to the kernel. These are stored so that they may be repeated
        on expressions the user specifies later.

    .. attribute:: cache_manager
    .. attribute:: isl_context
    .. attribute:: flags

        An instance of :class:`loopy.LoopyFlags`
    """

    # {{{ constructor

    def __init__(self, device, domains, instructions, args=[], schedule=None,
            name="loopy_kernel",
            preambles=[],
            preamble_generators=[default_preamble_generator],
            assumptions=None,
            local_sizes={},
            temporary_variables={},
            iname_to_tag={},
            substitutions={},
            function_manglers=[
                default_function_mangler,
                opencl_function_mangler,
                single_arg_function_mangler,
                ],
            symbol_manglers=[opencl_symbol_mangler],

            iname_slab_increments={},
            loop_priority=[],
            silenced_warnings=[],

            applied_iname_rewrites=[],
            cache_manager=None,
            index_dtype=np.int32,
            isl_context=None,
            flags=None,

            # When kernels get intersected in slab decomposition,
            # their grid sizes shouldn't change. This provides
            # a way to forward sub-kernel grid size requests.
            get_grid_sizes=None):

        if cache_manager is None:
            from loopy.kernel.tools import SetOperationCacheManager
            cache_manager = SetOperationCacheManager()

        # {{{ make instruction ids unique

        from loopy.kernel.creation import UniqueName

        insn_ids = set()
        for insn in instructions:
            if insn.id is not None and not isinstance(insn.id, UniqueName):
                if insn.id in insn_ids:
                    raise RuntimeError("duplicate instruction id: %s" % insn.id)
                insn_ids.add(insn.id)

        insn_id_gen = UniqueNameGenerator(insn_ids)

        new_instructions = []

        for insn in instructions:
            if insn.id is None:
                new_instructions.append(
                        insn.copy(id=insn_id_gen("insn")))
            elif isinstance(insn.id, UniqueName):
                new_instructions.append(
                        insn.copy(id=insn_id_gen(insn.id.name)))
            else:
                new_instructions.append(insn)

        instructions = new_instructions
        del new_instructions

        # }}}

        # {{{ process assumptions

        if assumptions is None:
            dom0_space = domains[0].get_space()
            assumptions_space = isl.Space.params_alloc(
                    dom0_space.get_ctx(), dom0_space.dim(dim_type.param))
            for i in xrange(dom0_space.dim(dim_type.param)):
                assumptions_space = assumptions_space.set_dim_name(
                        dim_type.param, i,
                        dom0_space.get_dim_name(dim_type.param, i))
            assumptions = isl.BasicSet.universe(assumptions_space)

        elif isinstance(assumptions, str):
            all_inames = set()
            all_params = set()
            for dom in domains:
                all_inames.update(dom.get_var_names(dim_type.set))
                all_params.update(dom.get_var_names(dim_type.param))

            domain_parameters = all_params-all_inames

            assumptions_set_str = "[%s] -> { : %s}" \
                    % (",".join(s for s in domain_parameters),
                        assumptions)
            assumptions = isl.BasicSet.read_from_str(domains[0].get_ctx(),
                    assumptions_set_str)

        assert assumptions.is_params()

        # }}}

        index_dtype = np.dtype(index_dtype)
        if index_dtype.kind != 'i':
            raise TypeError("index_dtype must be an integer")
        if np.iinfo(index_dtype).min >= 0:
            raise TypeError("index_dtype must be signed")

        if get_grid_sizes is not None:
            # overwrites method down below
            self.get_grid_sizes = get_grid_sizes

        Record.__init__(self,
                device=device, domains=domains,
                instructions=instructions,
                args=args,
                schedule=schedule,
                name=name,
                preambles=preambles,
                preamble_generators=preamble_generators,
                assumptions=assumptions,
                iname_slab_increments=iname_slab_increments,
                loop_priority=loop_priority,
                silenced_warnings=silenced_warnings,
                temporary_variables=temporary_variables,
                local_sizes=local_sizes,
                iname_to_tag=iname_to_tag,
                substitutions=substitutions,
                cache_manager=cache_manager,
                applied_iname_rewrites=applied_iname_rewrites,
                function_manglers=function_manglers,
                symbol_manglers=symbol_manglers,
                index_dtype=index_dtype,
                isl_context=isl_context,
                flags=flags)

    # }}}

    # {{{ function mangling

    def mangle_function(self, identifier, arg_dtypes):
        for mangler in self.function_manglers:
            mangle_result = mangler(identifier, arg_dtypes)
            if mangle_result is not None:
                return mangle_result

        return None

    # }}}

    # {{{ name wrangling

    @memoize_method
    def non_iname_variable_names(self):
        return (set(self.arg_dict.iterkeys())
                | set(self.temporary_variables.iterkeys()))

    @memoize_method
    def all_variable_names(self):
        return (
                set(self.temporary_variables.iterkeys())
                | set(self.substitutions.iterkeys())
                | set(arg.name for arg in self.args)
                | set(self.all_inames()))

    def get_var_name_generator(self):
        return _UniqueVarNameGenerator(self.all_variable_names())

    def make_unique_instruction_id(self, insns=None, based_on="insn",
            extra_used_ids=set()):
        if insns is None:
            insns = self.instructions

        used_ids = set(insn.id for insn in insns) | extra_used_ids

        for id_str in generate_unique_possibilities(based_on):
            if id_str not in used_ids:
                return id_str

    def get_var_descriptor(self, name):
        try:
            return self.arg_dict[name]
        except KeyError:
            pass

        try:
            return self.temporary_variables[name]
        except KeyError:
            pass

        raise ValueError("nothing known about variable '%s'" % name)

    @property
    @memoize_method
    def id_to_insn(self):
        return dict((insn.id, insn) for insn in self.instructions)

    # }}}

    # {{{ domain wrangling

    @memoize_method
    def parents_per_domain(self):
        """Return a list corresponding to self.domains (by index)
        containing domain indices which are nested around this
        domain.

        Each domains nest list walks from the leaves of the nesting
        tree to the root.
        """

        # The stack of iname sets records which inames are active
        # as we step through the linear list of domains. It also
        # determines the granularity of inames to be popped/decactivated
        # if we ascend a level.

        iname_set_stack = []
        result = []

        writer_map = self.writer_map()

        for dom in self.domains:
            parameters = set(dom.get_var_names(dim_type.param))
            inames = set(dom.get_var_names(dim_type.set))

            # This next domain may be nested inside the previous domain.
            # Or it may not, in which case we need to figure out how many
            # levels of parents we need to discard in order to find the
            # true parent.

            discard_level_count = 0
            while discard_level_count < len(iname_set_stack):
                # {{{ check for parenthood by loop bound iname

                last_inames = iname_set_stack[-1-discard_level_count]
                if last_inames & parameters:
                    break

                # }}}

                # {{{ check for parenthood by written variable

                is_parent_by_variable = False
                for par in parameters:
                    if par in self.temporary_variables:
                        writer_insns = writer_map[par]

                        if len(writer_insns) > 1:
                            raise RuntimeError("loop bound '%s' "
                                    "may only be written to once" % par)

                        writer_insn, = writer_insns
                        writer_inames = self.insn_inames(writer_insn)

                        if writer_inames & last_inames:
                            is_parent_by_variable = True
                            break

                if is_parent_by_variable:
                    break

                # }}}

                discard_level_count += 1

            if discard_level_count:
                iname_set_stack = iname_set_stack[:-discard_level_count]

            if result:
                parent = len(result)-1
            else:
                parent = None

            for i in range(discard_level_count):
                assert parent is not None
                parent = result[parent]

            # found this domain's parent
            result.append(parent)

            if iname_set_stack:
                parent_inames = iname_set_stack[-1]
            else:
                parent_inames = set()
            iname_set_stack.append(parent_inames | inames)

        return result

    @memoize_method
    def all_parents_per_domain(self):
        """Return a list corresponding to self.domains (by index)
        containing domain indices which are nested around this
        domain.

        Each domains nest list walks from the leaves of the nesting
        tree to the root.
        """
        result = []

        ppd = self.parents_per_domain()
        for dom, parent in zip(self.domains, ppd):
            # keep walking up tree to find *all* parents
            dom_result = []
            while parent is not None:
                dom_result.insert(0, parent)
                parent = ppd[parent]

            result.append(dom_result)

        return result

    @memoize_method
    def _get_home_domain_map(self):
        return dict(
                (iname, i_domain)
                for i_domain, dom in enumerate(self.domains)
                for iname in dom.get_var_names(dim_type.set))

    def get_home_domain_index(self, iname):
        return self._get_home_domain_map()[iname]

    @memoize_method
    def combine_domains(self, domains):
        """
        :arg domains: domain indices of domains to be combined. More 'dominant'
            domains (those which get most say on the actual dim_type of an iname)
            must be later in the order.
        """
        assert isinstance(domains, tuple)  # for caching

        if not domains:
            return isl.BasicSet.universe(isl.Space.set_alloc(
                self.isl_context, 0, 0))

        result = None
        for dom_index in domains:
            dom = self.domains[dom_index]
            if result is None:
                result = dom
            else:
                aligned_dom, aligned_result = isl.align_two(
                        dom, result, across_dim_types=True)
                result = aligned_result & aligned_dom

        return result

    def get_inames_domain(self, inames):
        if not inames:
            return self.combine_domains(())

        if isinstance(inames, str):
            inames = frozenset([inames])
        if not isinstance(inames, frozenset):
            inames = frozenset(inames)

            from warnings import warn
            warn("get_inames_domain did not get a frozenset", stacklevel=2)

        return self._get_inames_domain_backend(inames)

    @memoize_method
    def get_leaf_domain_indices(self, inames):
        """Find the leaves of the domain tree needed to cover all inames."""

        hdm = self._get_home_domain_map()
        ppd = self.all_parents_per_domain()

        domain_indices = set()

        # map root -> leaf
        root_to_leaf = {}

        for iname in inames:
            home_domain_index = hdm[iname]
            if home_domain_index in domain_indices:
                # nothin' new
                continue

            domain_parents = [home_domain_index] + ppd[home_domain_index]
            current_root = domain_parents[-1]
            previous_leaf = root_to_leaf.get(current_root)

            if previous_leaf is not None:
                # Check that we don't branch the domain tree.
                #
                # Branching the domain tree is dangerous/ill-formed because
                # it can introduce artificial restrictions on variables
                # further up the tree.

                prev_parents = set(ppd[previous_leaf])
                if not prev_parents <= set(domain_parents):
                    raise CannotBranchDomainTree("iname set '%s' requires "
                            "branch in domain tree (when adding '%s')"
                            % (", ".join(inames), iname))
            else:
                # We're adding a new root. That's fine.
                pass

            root_to_leaf[current_root] = home_domain_index
            domain_indices.update(domain_parents)

        return root_to_leaf.values()

    @memoize_method
    def _get_inames_domain_backend(self, inames):
        domain_indices = set()
        for leaf_dom_idx in self.get_leaf_domain_indices(inames):
            domain_indices.add(leaf_dom_idx)
            domain_indices.update(self.all_parents_per_domain()[leaf_dom_idx])

        return self.combine_domains(tuple(sorted(domain_indices)))

    # }}}

    # {{{ iname wrangling

    @memoize_method
    def all_inames(self):
        result = set()
        for dom in self.domains:
            result.update(dom.get_var_names(dim_type.set))
        return frozenset(result)

    @memoize_method
    def all_params(self):
        all_inames = self.all_inames()

        result = set()
        for dom in self.domains:
            result.update(set(dom.get_var_names(dim_type.param)) - all_inames)

        return frozenset(result)

    @memoize_method
    def all_insn_inames(self):
        """Return a mapping from instruction ids to inames inside which
        they should be run.
        """

        from loopy.kernel.tools import find_all_insn_inames
        return find_all_insn_inames(self)

    @memoize_method
    def all_referenced_inames(self):
        result = set()
        for inames in self.all_insn_inames().itervalues():
            result.update(inames)
        return result

    def insn_inames(self, insn):
        from loopy.kernel.data import InstructionBase
        if isinstance(insn, InstructionBase):
            return self.all_insn_inames()[insn.id]
        else:
            return self.all_insn_inames()[insn]

    @memoize_method
    def iname_to_insns(self):
        result = dict(
                (iname, set()) for iname in self.all_inames())
        for insn in self.instructions:
            for iname in self.insn_inames(insn):
                result[iname].add(insn.id)

        return result

    # }}}

    # {{{ read and written variables

    @memoize_method
    def reader_map(self):
        """
        :return: a dict that maps variable names to ids of insns that read that
          variable.
        """
        result = {}

        admissible_vars = (
                set(arg.name for arg in self.args)
                | set(self.temporary_variables.iterkeys()))

        for insn in self.instructions:
            for var_name in insn.read_dependency_names() & admissible_vars:
                result.setdefault(var_name, set()).add(insn.id)

    @memoize_method
    def writer_map(self):
        """
        :return: a dict that maps variable names to ids of insns that write
            to that variable.
        """
        result = {}

        for insn in self.instructions:
            for var_name, _ in insn.assignees_and_indices():
                result.setdefault(var_name, set()).add(insn.id)

        return result

    @memoize_method
    def get_read_variables(self):
        result = set()
        for insn in self.instructions:
            result.update(insn.read_dependency_names())
        return result

    @memoize_method
    def get_written_variables(self):
        return frozenset(
                var_name
                for insn in self.instructions
                for var_name, _ in insn.assignees_and_indices())

    # }}}

    # {{{ argument wrangling

    @property
    @memoize_method
    def arg_dict(self):
        return dict((arg.name, arg) for arg in self.args)

    @property
    @memoize_method
    def scalar_loop_args(self):
        from loopy.kernel.data import ValueArg

        if self.args is None:
            return []
        else:
            from pytools import flatten
            loop_arg_names = list(flatten(dom.get_var_names(dim_type.param)
                    for dom in self.domains))
            return [arg.name for arg in self.args if isinstance(arg, ValueArg)
                    if arg.name in loop_arg_names]
    # }}}

    # {{{ bounds finding

    @memoize_method
    def get_iname_bounds(self, iname):
        domain = self.get_inames_domain(frozenset([iname]))
        d_var_dict = domain.get_var_dict()

        assumptions = self.assumptions.project_out_except(
                set(domain.get_var_dict(dim_type.param)), [dim_type.param])

        aligned_assumptions, domain = isl.align_two(assumptions, domain)

        dom_intersect_assumptions = aligned_assumptions & domain

        lower_bound_pw_aff = (
                self.cache_manager.dim_min(
                    dom_intersect_assumptions,
                    d_var_dict[iname][1])
                .coalesce())
        upper_bound_pw_aff = (
                self.cache_manager.dim_max(
                    dom_intersect_assumptions,
                    d_var_dict[iname][1])
                .coalesce())

        class BoundsRecord(Record):
            pass

        size = (upper_bound_pw_aff - lower_bound_pw_aff + 1)
        size = size.gist(assumptions)

        return BoundsRecord(
                lower_bound_pw_aff=lower_bound_pw_aff,
                upper_bound_pw_aff=upper_bound_pw_aff,
                size=size)

    def find_var_base_indices_and_shape_from_inames(
            self, inames, cache_manager, context=None):
        if not inames:
            return [], []

        base_indices_and_sizes = [
                cache_manager.base_index_and_length(
                    self.get_inames_domain(iname), iname, context)
                for iname in inames]
        return zip(*base_indices_and_sizes)

    @memoize_method
    def get_constant_iname_length(self, iname):
        from loopy.isl_helpers import static_max_of_pw_aff
        from loopy.symbolic import aff_to_expr
        return int(aff_to_expr(static_max_of_pw_aff(
                self.get_iname_bounds(iname).size,
                constants_only=True)))

    @memoize_method
    def get_grid_sizes(self, ignore_auto=False):
        all_inames_by_insns = set()
        for insn in self.instructions:
            all_inames_by_insns |= self.insn_inames(insn)

        if not all_inames_by_insns <= self.all_inames():
            raise RuntimeError("some inames collected from instructions (%s) "
                    "are not present in domain (%s)"
                    % (", ".join(sorted(all_inames_by_insns)),
                        ", ".join(sorted(self.all_inames()))))

        global_sizes = {}
        local_sizes = {}

        from loopy.kernel.data import (
                GroupIndexTag, LocalIndexTag,
                AutoLocalIndexTagBase)

        for iname in self.all_inames():
            tag = self.iname_to_tag.get(iname)

            if isinstance(tag, GroupIndexTag):
                tgt_dict = global_sizes
            elif isinstance(tag, LocalIndexTag):
                tgt_dict = local_sizes
            elif isinstance(tag, AutoLocalIndexTagBase) and not ignore_auto:
                raise RuntimeError("cannot find grid sizes if automatic "
                        "local index tags are present")
            else:
                tgt_dict = None

            if tgt_dict is None:
                continue

            size = self.get_iname_bounds(iname).size

            if tag.axis in tgt_dict:
                size = tgt_dict[tag.axis].max(size)

            from loopy.isl_helpers import static_max_of_pw_aff
            try:
                # insist block size is constant
                size = static_max_of_pw_aff(size,
                        constants_only=isinstance(tag, LocalIndexTag))
            except ValueError:
                pass

            tgt_dict[tag.axis] = size

        max_dims = self.device.max_work_item_dimensions

        def to_dim_tuple(size_dict, which, forced_sizes={}):
            forced_sizes = forced_sizes.copy()

            size_list = []
            sorted_axes = sorted(size_dict.iterkeys())

            while sorted_axes or forced_sizes:
                if sorted_axes:
                    cur_axis = sorted_axes.pop(0)
                else:
                    cur_axis = None

                if len(size_list) in forced_sizes:
                    size_list.append(forced_sizes.pop(len(size_list)))
                    continue

                assert cur_axis is not None

                if cur_axis > len(size_list):
                    raise RuntimeError("%s axis %d unused" % (
                        which, len(size_list)))

                size_list.append(size_dict[cur_axis])

            if len(size_list) > max_dims:
                raise ValueError("more %s dimensions assigned than supported "
                        "by hardware (%d > %d)" % (which, len(size_list), max_dims))

            return tuple(size_list)

        return (to_dim_tuple(global_sizes, "global"),
                to_dim_tuple(local_sizes, "local", forced_sizes=self.local_sizes))

    def get_grid_sizes_as_exprs(self, ignore_auto=False):
        grid_size, group_size = self.get_grid_sizes(ignore_auto)

        def tup_to_exprs(tup):
            from loopy.symbolic import pw_aff_to_expr
            return tuple(pw_aff_to_expr(i, int_ok=True) for i in tup)

        return tup_to_exprs(grid_size), tup_to_exprs(group_size)

    # }}}

    # {{{ local memory

    @memoize_method
    def local_var_names(self):
        return set(
                tv.name
            for tv in self.temporary_variables.itervalues()
            if tv.is_local)

    def local_mem_use(self):
        return sum(lv.nbytes for lv in self.temporary_variables.itervalues()
                if lv.is_local)

    # }}}

    # {{{ pretty-printing

    def __str__(self):
        lines = []

        sep = 75*"-"
        lines.append(sep)
        lines.append("ARGUMENTS:")
        for arg_name in sorted(self.arg_dict):
            lines.append(str(self.arg_dict[arg_name]))
        lines.append(sep)
        lines.append("DOMAINS:")
        for dom, parents in zip(self.domains, self.all_parents_per_domain()):
            lines.append(len(parents)*"  " + str(dom))

        lines.append(sep)
        lines.append("INAME-TO-TAG MAP:")
        for iname in sorted(self.all_inames()):
            line = "%s: %s" % (iname, self.iname_to_tag.get(iname))
            lines.append(line)

        if self.substitutions:
            lines.append(sep)
            lines.append("SUBSTIUTION RULES:")
            for rule_name in sorted(self.substitutions.iterkeys()):
                lines.append(str(self.substitutions[rule_name]))

        if self.temporary_variables:
            lines.append(sep)
            lines.append("TEMPORARIES:")
            for tv in sorted(self.temporary_variables.itervalues(),
                    key=lambda tv: tv.name):
                lines.append(str(tv))

        lines.append(sep)
        lines.append("INSTRUCTIONS:")
        loop_list_width = 35

        import loopy as lp
        for insn in self.instructions:
            if isinstance(insn, lp.ExpressionInstruction):
                lhs = str(insn.assignee)
                rhs = str(insn.expression)
                trailing = []
            elif isinstance(insn, lp.CInstruction):
                lhs = ", ".join(str(a) for a in insn.assignees)
                rhs = "CODE(%s|%s)" % (
                        ", ".join(str(x) for x in insn.read_variables),
                        ", ".join("%s=%s" % (name, expr)
                            for name, expr in insn.iname_exprs))

                trailing = ["    "+l for l in insn.code.split("\n")]

            loop_list = ",".join(sorted(self.insn_inames(insn)))

            options = [insn.id]
            if insn.priority:
                options.append("priority=%d" % insn.priority)

            if len(loop_list) > loop_list_width:
                lines.append("[%s]" % loop_list)
                lines.append("%s%s <- %s   # %s" % (
                    (loop_list_width+2)*" ", lhs,
                    rhs, ", ".join(options)))
            else:
                lines.append("[%s]%s%s <- %s   # %s" % (
                    loop_list, " "*(loop_list_width-len(loop_list)),
                    lhs, rhs, ", ".join(options)))

            lines.extend(trailing)

        dep_lines = []
        for insn in self.instructions:
            if insn.insn_deps:
                dep_lines.append("%s : %s" % (insn.id, ",".join(insn.insn_deps)))
        if dep_lines:
            lines.append(sep)
            lines.append("DEPENDENCIES:")
            lines.extend(dep_lines)

        lines.append(sep)

        if self.schedule is not None:
            lines.append("SCHEDULE:")
            from loopy.schedule import dump_schedule
            lines.append(dump_schedule(self.schedule))
            lines.append(sep)

        return "\n".join(lines)

    # }}}

    # {{{ implementation arguments

    @property
    @memoize_method
    def impl_arg_to_arg(self):
        from loopy.kernel.array import ArrayBase

        result = {}

        for arg in self.args:
            if not isinstance(arg, ArrayBase):
                result[arg.name] = arg
                continue

            if arg.shape is None or arg.dim_tags is None:
                result[arg.name] = arg
                continue

            subscripts_and_names = arg.subscripts_and_names()
            if subscripts_and_names is None:
                result[arg.name] = arg
                continue

            for index, sub_arg_name in subscripts_and_names:
                result[sub_arg_name] = arg

        return result

    # }}}

    # {{{ direct execution

    @memoize_method
    def get_compiled_kernel(self, ctx, options, flags):
        from loopy.compiled import CompiledKernel
        return CompiledKernel(ctx, self, options=options, flags=flags)

    def __call__(self, queue, **kwargs):
        flags = kwargs.pop("flags", None)
        options = kwargs.pop("options", ())

        assert isinstance(options, tuple)

        return self.get_compiled_kernel(queue.context, options, flags)(
                queue, **kwargs)

    # }}}

# }}}

# vim: foldmethod=marker

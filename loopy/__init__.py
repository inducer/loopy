from __future__ import division, absolute_import

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


import six
from six.moves import range, zip

import islpy as isl

from loopy.symbolic import (RuleAwareIdentityMapper, RuleAwareSubstitutionMapper,
        SubstitutionRuleMappingContext,
        TaggedVariable, Reduction, LinearSubscript, )
from loopy.diagnostic import LoopyError, LoopyWarning


# {{{ imported user interface

from loopy.library.function import (
        default_function_mangler, single_arg_function_mangler)

from loopy.kernel.data import (
        auto,
        ValueArg, GlobalArg, ConstantArg, ImageArg,
        InstructionBase, ExpressionInstruction, CInstruction,
        TemporaryVariable)

from loopy.kernel import LoopKernel
from loopy.kernel.tools import (
        get_dot_dependency_graph,
        show_dependency_graph,
        add_dtypes,
        add_and_infer_dtypes)
from loopy.kernel.creation import make_kernel, UniqueName
from loopy.library.reduction import register_reduction_parser

# {{{ import transforms

from loopy.transform.iname import (
        assume, set_loop_priority,
        split_iname, join_inames, tag_inames, duplicate_inames,
        rename_iname, link_inames, remove_unused_inames,
        affine_map_inames)

from loopy.transform.instruction import (
        find_instructions, map_instructions,
        set_instruction_priority, add_dependency,
        remove_instructions, tag_instructions)

from loopy.transform.data import (
        add_prefetch, change_arg_to_image, tag_data_axes,
        set_array_dim_names, remove_unused_arguments,
        alias_temporaries, set_argument_order
        )

from loopy.transform.subst import (extract_subst,
        assignment_to_subst, expand_subst, find_rules_matching,
        find_one_rule_matching)

from loopy.transform.precompute import precompute
from loopy.transform.buffer import buffer_array
from loopy.transform.fusion import fuse_kernels

from loopy.transform.arithmetic import (
        split_reduction_inward,
        split_reduction_outward, fold_constants)

from loopy.transform.padding import (
        split_array_dim, split_arg_axis, find_padding_multiple,
        add_padding)

# }}}

from loopy.preprocess import (preprocess_kernel, realize_reduction,
        infer_unknown_types)
from loopy.schedule import generate_loop_schedules, get_one_scheduled_kernel
from loopy.statistics import (get_op_poly, get_gmem_access_poly,
        get_DRAM_access_poly, get_barrier_poly, stringify_stats_mapping,
        sum_mem_access_to_bytes)
from loopy.codegen import generate_code, generate_body
from loopy.compiled import CompiledKernel
from loopy.options import Options
from loopy.auto_test import auto_test_vs_ref
from loopy.frontend.fortran import (c_preprocess, parse_transformed_fortran,
        parse_fortran)

__all__ = [
        "TaggedVariable", "Reduction", "LinearSubscript",

        "auto",

        "LoopKernel",

        "ValueArg", "ScalarArg", "GlobalArg", "ArrayArg", "ConstantArg", "ImageArg",
        "TemporaryVariable",

        "InstructionBase", "ExpressionInstruction", "CInstruction",

        "default_function_mangler", "single_arg_function_mangler",

        "make_kernel", "UniqueName",

        "register_reduction_parser",

        # {{{ transforms

        "assume", "set_loop_priority",
        "split_iname", "join_inames", "tag_inames", "duplicate_inames",
        "rename_iname", "link_inames", "remove_unused_inames",
        "affine_map_inames",

        "add_prefetch", "change_arg_to_image", "tag_data_axes",
        "set_array_dim_names", "remove_unused_arguments",
        "alias_temporaries", "set_argument_order",

        "find_instructions", "map_instructions",
        "set_instruction_priority", "add_dependency",
        "remove_instructions", "tag_instructions",

        "extract_subst", "expand_subst", "assignment_to_subst",
        "find_rules_matching", "find_one_rule_matching",

        "precompute", "buffer_array",
        "fuse_kernels",

        "split_reduction_inward", "split_reduction_outward",
        "fold_constants",

        "split_array_dim", "split_arg_axis", "find_padding_multiple",
        "add_padding",

        # }}}

        "get_dot_dependency_graph",
        "show_dependency_graph",
        "add_dtypes",
        "infer_argument_dtypes", "add_and_infer_dtypes",

        "preprocess_kernel", "realize_reduction", "infer_unknown_types",
        "generate_loop_schedules", "get_one_scheduled_kernel",
        "generate_code", "generate_body",

        "get_op_poly", "get_gmem_access_poly", "get_DRAM_access_poly",
        "get_barrier_poly", "stringify_stats_mapping", "sum_mem_access_to_bytes",

        "CompiledKernel",

        "auto_test_vs_ref",

        "Options",

        "make_kernel",
        "c_preprocess", "parse_transformed_fortran", "parse_fortran",

        "LoopyError", "LoopyWarning",

        # {{{ from this file

        "fix_parameters",
        "register_preamble_generators",
        "register_symbol_manglers",
        "register_function_manglers",

        "set_caching_enabled",
        "CacheMode",
        "make_copy_kernel",
        "to_batched",
        "realize_ilp",

        # }}}
        ]


# }}}


# {{{ fix_parameter

def _fix_parameter(kernel, name, value):
    def process_set(s):
        var_dict = s.get_var_dict()

        try:
            dt, idx = var_dict[name]
        except KeyError:
            return s

        value_aff = isl.Aff.zero_on_domain(s.space) + value

        from loopy.isl_helpers import iname_rel_aff
        name_equal_value_aff = iname_rel_aff(s.space, name, "==", value_aff)

        s = (s
                .add_constraint(
                    isl.Constraint.equality_from_aff(name_equal_value_aff))
                .project_out(dt, idx, 1))

        return s

    new_domains = [process_set(dom) for dom in kernel.domains]

    from pymbolic.mapper.substitutor import make_subst_func
    subst_func = make_subst_func({name: value})

    from loopy.symbolic import SubstitutionMapper, PartialEvaluationMapper
    subst_map = SubstitutionMapper(subst_func)
    ev_map = PartialEvaluationMapper()

    def map_expr(expr):
        return ev_map(subst_map(expr))

    from loopy.kernel.array import ArrayBase
    new_args = []
    for arg in kernel.args:
        if arg.name == name:
            # remove from argument list
            continue

        if not isinstance(arg, ArrayBase):
            new_args.append(arg)
        else:
            new_args.append(arg.map_exprs(map_expr))

    new_temp_vars = {}
    for tv in six.itervalues(kernel.temporary_variables):
        new_temp_vars[tv.name] = tv.map_exprs(map_expr)

    from loopy.context_matching import parse_stack_match
    within = parse_stack_match(None)

    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())
    esubst_map = RuleAwareSubstitutionMapper(
            rule_mapping_context, subst_func, within=within)
    return (
            rule_mapping_context.finish_kernel(
                esubst_map.map_kernel(kernel))
            .copy(
                domains=new_domains,
                args=new_args,
                temporary_variables=new_temp_vars,
                assumptions=process_set(kernel.assumptions),
                ))


def fix_parameters(kernel, **value_dict):
    for name, value in six.iteritems(value_dict):
        kernel = _fix_parameter(kernel, name, value)

    return kernel

# }}}


# {{{ set_options

def set_options(kernel, *args, **kwargs):
    """Return a new kernel with the options given as keyword arguments, or from
    a string representation passed in as the first (and only) positional
    argument.

    See also :class:`Options`.
    """

    if args and kwargs:
        raise TypeError("cannot pass both positional and keyword arguments")

    new_opt = kernel.options.copy()

    if kwargs:
        for key, val in six.iteritems(kwargs):
            if not hasattr(new_opt, key):
                raise ValueError("unknown option '%s'" % key)

            setattr(new_opt, key, val)
    else:
        if len(args) != 1:
            raise TypeError("exactly one positional argument is required if "
                    "no keyword args are given")
        arg, = args

        from loopy.options import make_options
        new_opt.update(make_options(arg))

    return kernel.copy(options=new_opt)

# }}}


# {{{ library registration

def register_preamble_generators(kernel, preamble_generators):
    new_pgens = kernel.preamble_generators[:]
    for pgen in preamble_generators:
        if pgen not in new_pgens:
            new_pgens.insert(0, pgen)

    return kernel.copy(preamble_generators=new_pgens)


def register_symbol_manglers(kernel, manglers):
    new_manglers = kernel.symbol_manglers[:]
    for m in manglers:
        if m not in new_manglers:
            new_manglers.insert(0, m)

    return kernel.copy(symbol_manglers=new_manglers)


def register_function_manglers(kernel, manglers):
    new_manglers = kernel.function_manglers[:]
    for m in manglers:
        if m not in new_manglers:
            new_manglers.insert(0, m)

    return kernel.copy(function_manglers=new_manglers)

# }}}


# {{{ cache control

import os
CACHING_ENABLED = (
    "LOOPY_NO_CACHE" not in os.environ
    and
    "CG_NO_CACHE" not in os.environ)


def set_caching_enabled(flag):
    """Set whether :mod:`loopy` is allowed to use disk caching for its various
    code generation stages.
    """
    global CACHING_ENABLED
    CACHING_ENABLED = flag


class CacheMode(object):
    """A context manager for setting whether :mod:`loopy` is allowed to use
    disk caches.
    """

    def __init__(self, new_flag):
        self.new_flag = new_flag

    def __enter__(self):
        global CACHING_ENABLED
        self.previous_mode = CACHING_ENABLED
        CACHING_ENABLED = self.new_flag

    def __exit__(self, exc_type, exc_val, exc_tb):
        global CACHING_ENABLED
        CACHING_ENABLED = self.previous_mode
        del self.previous_mode

# }}}


# {{{ make copy kernel

def make_copy_kernel(new_dim_tags, old_dim_tags=None):
    """Returns a :class:`LoopKernel` that changes the data layout
    of a variable (called "input") to the new layout specified by
    *new_dim_tags* from the one specified by *old_dim_tags*.
    *old_dim_tags* defaults to an all-C layout of the same rank
    as the one given by *new_dim_tags*.
    """

    from loopy.kernel.array import (parse_array_dim_tags,
            SeparateArrayArrayDimTag, VectorArrayDimTag)
    new_dim_tags = parse_array_dim_tags(new_dim_tags, n_axes=None)

    rank = len(new_dim_tags)
    if old_dim_tags is None:
        old_dim_tags = parse_array_dim_tags(
                ",".join(rank * ["c"]), n_axes=None)
    elif isinstance(old_dim_tags, str):
        old_dim_tags = parse_array_dim_tags(
                old_dim_tags, n_axes=None)

    indices = ["i%d" % i for i in range(rank)]
    shape = ["n%d" % i for i in range(rank)]
    commad_indices = ", ".join(indices)
    bounds = " and ".join(
            "0<=%s<%s" % (ind, shape_i)
            for ind, shape_i in zip(indices, shape))

    set_str = "{[%s]: %s}" % (
                commad_indices,
                bounds
                )
    result = make_kernel(set_str,
            "output[%s] = input[%s]"
            % (commad_indices, commad_indices))

    result = tag_data_axes(result, "input", old_dim_tags)
    result = tag_data_axes(result, "output", new_dim_tags)

    unrolled_tags = (SeparateArrayArrayDimTag, VectorArrayDimTag)
    for i in range(rank):
        if (isinstance(new_dim_tags[i], unrolled_tags)
                or isinstance(old_dim_tags[i], unrolled_tags)):
            result = tag_inames(result, {indices[i]: "unr"})

    return result

# }}}


# {{{ to_batched

class _BatchVariableChanger(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, kernel, batch_varying_args,
            batch_iname_expr):
        super(_BatchVariableChanger, self).__init__(rule_mapping_context)

        self.kernel = kernel
        self.batch_varying_args = batch_varying_args
        self.batch_iname_expr = batch_iname_expr

    def needs_batch_subscript(self, name):
        return (
                name in self.kernel.temporary_variables
                or
                name in self.batch_varying_args)

    def map_subscript(self, expr, expn_state):
        if not self.needs_batch_subscript(expr.aggregate.name):
            return super(_BatchVariableChanger, self).map_subscript(expr, expn_state)

        idx = expr.index
        if not isinstance(idx, tuple):
            idx = (idx,)

        return type(expr)(expr.aggregate, (self.batch_iname_expr,) + idx)

    def map_variable(self, expr, expn_state):
        if not self.needs_batch_subscript(expr.name):
            return super(_BatchVariableChanger, self).map_variable(expr, expn_state)

        return expr.aggregate[self.batch_iname_expr]


def to_batched(knl, nbatches, batch_varying_args, batch_iname_prefix="ibatch"):
    """Takes in a kernel that carries out an operation and returns a kernel
    that carries out a batch of these operations.

    :arg nbatches: the number of batches. May be a constant non-negative
        integer or a string, which will be added as an integer argument.
    :arg batch_varying_args: a list of argument names that depend vary per-batch.
        Each such variable will have a batch index added.
    """

    from pymbolic import var

    vng = knl.get_var_name_generator()
    batch_iname = vng(batch_iname_prefix)
    batch_iname_expr = var(batch_iname)

    new_args = []

    batch_dom_str = "{[%(iname)s]: 0 <= %(iname)s < %(nbatches)s}" % {
            "iname": batch_iname,
            "nbatches": nbatches,
            }

    if not isinstance(nbatches, int):
        batch_dom_str = "[%s] -> " % nbatches + batch_dom_str
        new_args.append(ValueArg(nbatches, dtype=knl.index_dtype))

        nbatches_expr = var(nbatches)
    else:
        nbatches_expr = nbatches

    batch_domain = isl.BasicSet(batch_dom_str)
    new_domains = [batch_domain] + knl.domains

    for arg in knl.args:
        if arg.name in batch_varying_args:
            if isinstance(arg, ValueArg):
                arg = GlobalArg(arg.name, arg.dtype, shape=(nbatches_expr,),
                        dim_tags="c")
            else:
                arg = arg.copy(
                        shape=(nbatches_expr,) + arg.shape,
                        dim_tags=("c",) * (len(arg.shape) + 1))

        new_args.append(arg)

    new_temps = {}

    for temp in six.itervalues(knl.temporary_variables):
        new_temps[temp.name] = temp.copy(
                shape=(nbatches_expr,) + temp.shape,
                dim_tags=("c",) * (len(arg.shape) + 1))

    knl = knl.copy(
            domains=new_domains,
            args=new_args,
            temporary_variables=new_temps)

    rule_mapping_context = SubstitutionRuleMappingContext(
            knl.substitutions, vng)
    bvc = _BatchVariableChanger(rule_mapping_context,
            knl, batch_varying_args, batch_iname_expr)
    return rule_mapping_context.finish_kernel(
            bvc.map_kernel(knl))


# }}}


# {{{ realize_ilp

def realize_ilp(kernel, iname):
    """Instruction-level parallelism (as realized by the loopy iname
    tag ``"ilp"``) provides the illusion that multiple concurrent
    program instances execute in lockstep within a single instruction
    stream.

    To do so, storage that is private to each instruction stream needs to be
    duplicated so that each program instance receives its own copy.  Storage
    that is written to in an instruction using an ILP iname but whose left-hand
    side indices do not contain said ILP iname is marked for duplication.

    This storage duplication is carried out automatically at code generation
    time, but, using this function, can also be carried out ahead of time
    on a per-iname basis (so that, for instance, data layout of the duplicated
    storage can be controlled explicitly.
    """
    from loopy.ilp import add_axes_to_temporaries_for_ilp_and_vec
    return add_axes_to_temporaries_for_ilp_and_vec(kernel, iname)

# }}}


# vim: foldmethod=marker

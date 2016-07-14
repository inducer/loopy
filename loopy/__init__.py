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

from loopy.symbolic import (
        TaggedVariable, Reduction, LinearSubscript, )
from loopy.diagnostic import LoopyError, LoopyWarning


# {{{ imported user interface

from loopy.library.function import (
        default_function_mangler, single_arg_function_mangler)

from loopy.kernel.data import (
        auto,
        KernelArgument,
        ValueArg, GlobalArg, ConstantArg, ImageArg,
        memory_ordering, memory_scope, VarAtomicity, AtomicInit, AtomicUpdate,
        InstructionBase,
        MultiAssignmentBase, Assignment, ExpressionInstruction,
        CallInstruction, CInstruction,
        temp_var_scope, TemporaryVariable,
        SubstitutionRule,
        CallMangleInfo)

from loopy.kernel import LoopKernel, kernel_state
from loopy.kernel.tools import (
        get_dot_dependency_graph,
        show_dependency_graph,
        add_dtypes,
        add_and_infer_dtypes)
from loopy.kernel.creation import make_kernel, UniqueName
from loopy.library.reduction import register_reduction_parser

# {{{ import transforms

from loopy.transform.iname import (
        set_loop_priority,
        split_iname, chunk_iname, join_inames, tag_inames, duplicate_inames,
        rename_iname, remove_unused_inames,
        split_reduction_inward, split_reduction_outward,
        affine_map_inames, find_unused_axis_tag,
        make_reduction_inames_unique)

from loopy.transform.instruction import (
        find_instructions, map_instructions,
        set_instruction_priority, add_dependency,
        remove_instructions,
        replace_instruction_ids,
        tag_instructions)

from loopy.transform.data import (
        add_prefetch, change_arg_to_image,
        tag_array_axes, tag_data_axes,
        set_array_axis_names, set_array_dim_names,
        remove_unused_arguments,
        alias_temporaries, set_argument_order,
        rename_argument,
        set_temporary_scope)

from loopy.transform.subst import (extract_subst,
        assignment_to_subst, expand_subst, find_rules_matching,
        find_one_rule_matching)

from loopy.transform.precompute import precompute
from loopy.transform.buffer import buffer_array
from loopy.transform.fusion import fuse_kernels

from loopy.transform.arithmetic import (
        fold_constants,
        collect_common_factors_on_increment)

from loopy.transform.padding import (
        split_array_axis, split_array_dim, split_arg_axis,
        find_padding_multiple,
        add_padding)

from loopy.transform.ilp import realize_ilp
from loopy.transform.batch import to_batched
from loopy.transform.parameter import assume, fix_parameters

# }}}

from loopy.preprocess import (preprocess_kernel, realize_reduction,
        infer_unknown_types)
from loopy.schedule import generate_loop_schedules, get_one_scheduled_kernel
from loopy.statistics import (get_op_poly, sum_ops_to_dtypes,
        get_gmem_access_poly,
        get_DRAM_access_poly, get_synchronization_poly, stringify_stats_mapping,
        sum_mem_access_to_bytes,
        gather_access_footprints, gather_access_footprint_bytes)
from loopy.codegen import (
        generate_code, generate_code_v2, generate_body)
from loopy.codegen.result import (
        GeneratedProgram,
        CodeGenerationResult)
from loopy.compiled import CompiledKernel
from loopy.options import Options
from loopy.auto_test import auto_test_vs_ref
from loopy.frontend.fortran import (c_preprocess, parse_transformed_fortran,
        parse_fortran)

from loopy.target import TargetBase, ASTBuilderBase
from loopy.target.c import CTarget
from loopy.target.cuda import CudaTarget
from loopy.target.opencl import OpenCLTarget
from loopy.target.pyopencl import PyOpenCLTarget
from loopy.target.ispc import ISPCTarget
from loopy.target.numba import NumbaTarget, NumbaCudaTarget


__all__ = [
        "TaggedVariable", "Reduction", "LinearSubscript",

        "auto",

        "LoopKernel", "kernel_state",

        "KernelArgument",
        "memory_ordering", "memory_scope", "VarAtomicity",
        "AtomicInit", "AtomicUpdate",
        "ValueArg", "GlobalArg", "ConstantArg", "ImageArg",
        "temp_var_scope", "TemporaryVariable",
        "SubstitutionRule",
        "CallMangleInfo",

        "InstructionBase",
        "MultiAssignmentBase", "Assignment", "ExpressionInstruction",
        "CallInstruction", "CInstruction",

        "default_function_mangler", "single_arg_function_mangler",

        "make_kernel", "UniqueName",

        "register_reduction_parser",

        # {{{ transforms

        "set_loop_priority",
        "split_iname", "chunk_iname", "join_inames", "tag_inames",
        "duplicate_inames",
        "rename_iname", "remove_unused_inames",
        "split_reduction_inward", "split_reduction_outward",
        "affine_map_inames", "find_unused_axis_tag",
        "make_reduction_inames_unique",

        "add_prefetch", "change_arg_to_image",
        "tag_array_axes", "tag_data_axes",
        "set_array_axis_names", "set_array_dim_names",
        "remove_unused_arguments",
        "alias_temporaries", "set_argument_order",
        "rename_argument", "set_temporary_scope",

        "find_instructions", "map_instructions",
        "set_instruction_priority", "add_dependency",
        "remove_instructions",
        "replace_instruction_ids",
        "tag_instructions",

        "extract_subst", "expand_subst", "assignment_to_subst",
        "find_rules_matching", "find_one_rule_matching",

        "precompute", "buffer_array",
        "fuse_kernels",

        "fold_constants", "collect_common_factors_on_increment",

        "split_array_axis", "split_array_dim", "split_arg_axis",
        "find_padding_multiple", "add_padding",

        "realize_ilp",

        "to_batched",

        "assume", "fix_parameters",

        # }}}

        "get_dot_dependency_graph",
        "show_dependency_graph",
        "add_dtypes",
        "add_and_infer_dtypes",

        "preprocess_kernel", "realize_reduction", "infer_unknown_types",
        "generate_loop_schedules", "get_one_scheduled_kernel",
        "GeneratedProgram", "CodeGenerationResult",
        "generate_code", "generate_code_v2", "generate_body",

        "get_op_poly", "sum_ops_to_dtypes", "get_gmem_access_poly",
        "get_DRAM_access_poly",
        "get_synchronization_poly", "stringify_stats_mapping",
        "sum_mem_access_to_bytes",
        "gather_access_footprints", "gather_access_footprint_bytes",

        "CompiledKernel",

        "auto_test_vs_ref",

        "Options",

        "make_kernel",
        "c_preprocess", "parse_transformed_fortran", "parse_fortran",

        "LoopyError", "LoopyWarning",

        "TargetBase", "CTarget", "CudaTarget", "OpenCLTarget",
        "PyOpenCLTarget", "ISPCTarget",
        "NumbaTarget", "NumbaCudaTarget",
        "ASTBuilderBase",

        # {{{ from this file

        "register_preamble_generators",
        "register_symbol_manglers",
        "register_function_manglers",

        "set_caching_enabled",
        "CacheMode",
        "make_copy_kernel",

        # }}}
        ]


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
    """
    :arg manglers: list of functions of signature ``(target, name, arg_dtypes)``
        returning a :class:`loopy.CallMangleInfo`.
    :returns: *kernel* with *manglers* registered
    """
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

    result = tag_array_axes(result, "input", old_dim_tags)
    result = tag_array_axes(result, "output", new_dim_tags)

    unrolled_tags = (SeparateArrayArrayDimTag, VectorArrayDimTag)
    for i in range(rank):
        if (isinstance(new_dim_tags[i], unrolled_tags)
                or isinstance(old_dim_tags[i], unrolled_tags)):
            result = tag_inames(result, {indices[i]: "unr"})

    return result

# }}}


# {{{ default target

_DEFAULT_TARGET = None


def set_default_target(target):
    # deliberately undocumented for now
    global _DEFAULT_TARGET
    _DEFAULT_TARGET = target


def _set_up_default_target():
    try:
        import pyopencl  # noqa
    except ImportError:
        from loopy.target.opencl import OpenCLTarget
        target = OpenCLTarget()
    else:
        from loopy.target.pyopencl import PyOpenCLTarget
        target = PyOpenCLTarget()

    set_default_target(target)

_set_up_default_target()


# }}}


# vim: foldmethod=marker

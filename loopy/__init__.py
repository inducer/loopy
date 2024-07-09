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


from loopy.auto_test import auto_test_vs_ref
from loopy.codegen import PreambleInfo, generate_body, generate_code, generate_code_v2
from loopy.codegen.result import CodeGenerationResult, GeneratedProgram
from loopy.diagnostic import LoopyError, LoopyWarning
from loopy.frontend.fortran import (
    c_preprocess,
    parse_fortran,
    parse_transformed_fortran,
)
from loopy.kernel import KernelState, LoopKernel
from loopy.kernel.creation import UniqueName, make_function, make_kernel
from loopy.kernel.data import (
    AddressSpace,
    ArrayArg,
    CallMangleInfo,
    ConstantArg,
    GlobalArg,
    ImageArg,
    KernelArgument,
    SubstitutionRule,
    TemporaryVariable,
    ValueArg,
)
from loopy.kernel.function_interface import (
    CallableKernel,
    InKernelCallable,
    ScalarCallable,
)
from loopy.kernel.instruction import (
    Assignment,
    AtomicInit,
    AtomicUpdate,
    BarrierInstruction,
    CallInstruction,
    CInstruction,
    InstructionBase,
    LegacyStringInstructionTag,
    MemoryOrdering,
    MemoryScope,
    MultiAssignmentBase,
    NoOpInstruction,
    OrderedAtomic,
    UseStreamingStoreTag,
    VarAtomicity,
)
from loopy.kernel.tools import (
    add_and_infer_dtypes,
    add_dtypes,
    find_most_recent_global_barrier,
    get_dot_dependency_graph,
    get_global_barrier_order,
    get_subkernel_to_insn_id_map,
    get_subkernels,
    show_dependency_graph,
)
from loopy.library.reduction import register_reduction_parser
from loopy.options import Options
from loopy.preprocess import infer_arg_descr, preprocess_kernel, preprocess_program
from loopy.schedule import (
    generate_loop_schedules,
    get_one_linearized_kernel,
    get_one_scheduled_kernel,
    linearize,
)
from loopy.statistics import (
    CountGranularity,
    MemAccess,
    Op,
    Sync,
    ToCountMap,
    ToCountPolynomialMap,
    gather_access_footprint_bytes,
    gather_access_footprints,
    get_mem_access_map,
    get_op_map,
    get_synchronization_map,
)
from loopy.symbolic import LinearSubscript, Reduction, TaggedVariable, TypeCast
from loopy.target import ASTBuilderBase, TargetBase
from loopy.target.c import (
    CFamilyTarget,
    CTarget,
    CWithGNULibcTarget,
    ExecutableCTarget,
    ExecutableCWithGNULibcTarget,
    generate_header,
)
from loopy.target.cuda import CudaTarget
from loopy.target.execution import ExecutorBase
from loopy.target.ispc import ISPCTarget
from loopy.target.opencl import OpenCLTarget
from loopy.target.pyopencl import PyOpenCLTarget
from loopy.tools import Optional, clear_in_mem_caches, memoize_on_disk, t_unit_to_python
from loopy.transform.add_barrier import add_barrier
from loopy.transform.arithmetic import (
    collect_common_factors_on_increment,
    fold_constants,
)
from loopy.transform.batch import to_batched
from loopy.transform.buffer import buffer_array
from loopy.transform.callable import (
    inline_callable_kernel,
    merge,
    register_callable,
    rename_callable,
)
from loopy.transform.concatenate import concatenate_arrays
from loopy.transform.data import (
    add_prefetch,
    alias_temporaries,
    allocate_temporaries_for_base_storage,
    change_arg_to_image,
    remove_unused_arguments,
    rename_argument,
    set_argument_order,
    set_array_axis_names,
    set_array_dim_names,
    set_temporary_address_space,
    set_temporary_scope,
    tag_array_axes,
    tag_data_axes,
)
from loopy.transform.fusion import fuse_kernels
from loopy.transform.iname import (
    add_inames_for_unused_hw_axes,
    add_inames_to_insn,
    affine_map_inames,
    chunk_iname,
    duplicate_inames,
    find_unused_axis_tag,
    get_iname_duplication_options,
    has_schedulable_iname_nesting,
    join_inames,
    make_reduction_inames_unique,
    map_domain,
    prioritize_loops,
    remove_inames_from_insn,
    remove_predicates_from_insn,
    remove_unused_inames,
    rename_iname,
    rename_inames,
    split_iname,
    split_reduction_inward,
    split_reduction_outward,
    tag_inames,
    untag_inames,
)
from loopy.transform.instruction import (
    add_dependency,
    add_nosync,
    find_instructions,
    map_instructions,
    remove_instructions,
    replace_instruction_ids,
    set_instruction_priority,
    simplify_indices,
    tag_instructions,
)
from loopy.transform.pack_and_unpack_args import pack_and_unpack_args_for_call
from loopy.transform.padding import (
    add_padding,
    find_padding_multiple,
    split_arg_axis,
    split_array_axis,
    split_array_dim,
)
from loopy.transform.parameter import assume, fix_parameters
from loopy.transform.precompute import precompute
from loopy.transform.privatize import (
    privatize_temporaries_with_inames,
    unprivatize_temporaries_with_inames,
)
from loopy.transform.realize_reduction import realize_reduction
from loopy.transform.save import save_and_reload_temporaries
from loopy.transform.subst import (
    assignment_to_subst,
    expand_subst,
    extract_subst,
    find_one_rule_matching,
    find_rules_matching,
)
from loopy.translation_unit import TranslationUnit, for_each_kernel, make_program

# }}}
from loopy.type_inference import infer_unknown_types
from loopy.types import to_loopy_type

# {{{ imported user interface
from loopy.typing import auto

# {{{ import transforms
from loopy.version import MOST_RECENT_LANGUAGE_VERSION, VERSION


__all__ = [
    "MOST_RECENT_LANGUAGE_VERSION",
    "VERSION",
    "ASTBuilderBase",
    "AddressSpace",
    "ArrayArg",
    "Assignment",
    "AtomicInit",
    "AtomicUpdate",
    "BarrierInstruction",
    "CFamilyTarget",
    "CInstruction",
    "CTarget",
    "CWithGNULibcTarget",
    "CacheMode",
    "CallInstruction",
    "CallMangleInfo",
    "CallableKernel",
    "CodeGenerationResult",
    "ConstantArg",
    "CountGranularity",
    "CudaTarget",
    "ExecutableCTarget",
    "ExecutableCWithGNULibcTarget",
    "ExecutorBase",
    "GeneratedProgram",
    "GlobalArg",
    "ISPCTarget",
    "ImageArg",
    "InKernelCallable",
    "InstructionBase",
    "KernelArgument",
    "KernelState",
    "LegacyStringInstructionTag",
    "LinearSubscript",
    "LoopKernel",
    "LoopyError",
    "LoopyWarning",
    "MemAccess",
    "MemoryOrdering",
    "MemoryScope",
    "MultiAssignmentBase",
    "NoOpInstruction",
    "Op",
    "OpenCLTarget",
    "Optional",
    "Options",
    "OrderedAtomic",
    "PreambleInfo",
    "PyOpenCLTarget",
    "Reduction",
    "ScalarCallable",
    "SubstitutionRule",
    "Sync",
    "TaggedVariable",
    "TargetBase",
    "TemporaryVariable",
    "ToCountMap",
    "ToCountPolynomialMap",
    "TranslationUnit",
    "TypeCast",
    "UniqueName",
    "UseStreamingStoreTag",
    "ValueArg",
    "VarAtomicity",
    "add_and_infer_dtypes",
    "add_barrier",
    "add_dependency",
    "add_dtypes",
    "add_inames_for_unused_hw_axes",
    "add_inames_to_insn",
    "add_nosync",
    "add_padding",
    "add_prefetch",
    "affine_map_inames",
    "alias_temporaries",
    "allocate_temporaries_for_base_storage",
    "assignment_to_subst",
    "assume",
    "auto",
    "auto_test_vs_ref",
    "buffer_array",
    "c_preprocess",
    "change_arg_to_image",
    "chunk_iname",
    "clear_in_mem_caches",
    "collect_common_factors_on_increment",
    "concatenate_arrays",
    "duplicate_inames",
    "expand_subst",
    "extract_subst",
    "find_instructions",
    "find_most_recent_global_barrier",
    "find_one_rule_matching",
    "find_padding_multiple",
    "find_rules_matching",
    "find_unused_axis_tag",
    "fix_parameters",
    "fold_constants",
    "for_each_kernel",
    "fuse_kernels",
    "gather_access_footprint_bytes",
    "gather_access_footprints",
    "generate_body",
    "generate_code",
    "generate_code_v2",
    "generate_header",
    "generate_loop_schedules",
    "get_dot_dependency_graph",
    "get_global_barrier_order",
    "get_iname_duplication_options",
    "get_mem_access_map",
    "get_one_linearized_kernel",
    "get_one_scheduled_kernel",
    "get_op_map",
    "get_subkernel_to_insn_id_map",
    "get_subkernels",
    "get_synchronization_map",
    "has_schedulable_iname_nesting",
    "infer_arg_descr",
    "infer_unknown_types",
    "inline_callable_kernel",
    "join_inames",
    "linearize",
    "make_copy_kernel",
    "make_einsum",
    "make_function",
    "make_kernel",
    "make_kernel",
    "make_program",
    "make_reduction_inames_unique",
    "map_domain",
    "map_instructions",
    "memoize_on_disk",
    "merge",
    "pack_and_unpack_args_for_call",
    "parse_fortran",
    "parse_transformed_fortran",
    "precompute",
    "preprocess_kernel",
    "preprocess_program",
    "prioritize_loops",
    "privatize_temporaries_with_inames",
    "realize_reduction",
    "register_callable",
    "register_preamble_generators",
    "register_reduction_parser",
    "register_symbol_manglers",
    "remove_inames_from_insn",
    "remove_instructions",
    "remove_predicates_from_insn",
    "remove_unused_arguments",
    "remove_unused_inames",
    "rename_argument",
    "rename_callable",
    "rename_iname",
    "rename_inames",
    "replace_instruction_ids",
    "save_and_reload_temporaries",
    "set_argument_order",
    "set_array_axis_names",
    "set_array_dim_names",
    "set_caching_enabled",
    "set_instruction_priority",
    "set_temporary_address_space",
    "set_temporary_scope",
    "show_dependency_graph",
    "simplify_indices",
    "split_arg_axis",
    "split_array_axis",
    "split_array_dim",
    "split_iname",
    "split_reduction_inward",
    "split_reduction_outward",
    "t_unit_to_python",
    "tag_array_axes",
    "tag_data_axes",
    "tag_inames",
    "tag_instructions",
    "to_batched",
    "to_loopy_type",
    "unprivatize_temporaries_with_inames",
    "untag_inames",
    ]

# }}}


# {{{ set_options

@for_each_kernel
def set_options(kernel, *args, **kwargs):
    """Return a new kernel with the options given as keyword arguments, or from
    a string representation passed in as the first (and only) positional
    argument.

    See also :class:`Options`.
    """
    assert isinstance(kernel, LoopKernel)

    if args and kwargs:
        raise TypeError("cannot pass both positional and keyword arguments")

    new_opt = kernel.options.copy()

    if kwargs:
        from loopy.options import Options, _apply_legacy_map
        kwargs = _apply_legacy_map(Options._legacy_options_map, kwargs)

        for key, val in kwargs.items():
            if not hasattr(new_opt, key):
                raise ValueError("unknown option '%s'" % key)

            setattr(new_opt, key, val)
    else:
        if len(args) != 1:
            raise TypeError("exactly one positional argument is required if "
                    "no keyword args are given")
        arg, = args

        from loopy.options import make_options
        new_opt._update(make_options(arg))

    return kernel.copy(options=new_opt)

# }}}


# {{{ library registration

@for_each_kernel
def register_preamble_generators(kernel: LoopKernel, preamble_generators):
    """
    :arg manglers: list of functions of signature ``(preamble_info)``
        generating tuples ``(sortable_str_identifier, code)``,
        where *preamble_info* is a :class:`PreambleInfo`.

    :returns: *kernel* with *manglers* registered
    """
    from loopy.tools import unpickles_equally

    new_pgens = tuple(kernel.preamble_generators)

    for pgen in preamble_generators:
        if pgen not in new_pgens:
            if not unpickles_equally(pgen):
                raise LoopyError("preamble generator '%s' does not "
                        "compare equally after being unpickled "
                        "and would thus disrupt loopy's caches"
                        % pgen)

            new_pgens = (pgen,) + new_pgens

    return kernel.copy(preamble_generators=new_pgens)


@for_each_kernel
def register_symbol_manglers(kernel, manglers):
    from loopy.tools import unpickles_equally

    new_manglers = kernel.symbol_manglers
    for m in manglers:
        if m not in new_manglers:
            if not unpickles_equally(m):
                raise LoopyError("mangler '%s' does not "
                        "compare equally after being unpickled "
                        "and would disrupt loopy's caches"
                        % m)

            new_manglers = (m,) + new_manglers

    return kernel.copy(symbol_manglers=new_manglers)

# }}}


# {{{ cache control

import os

from pytools import strtobool


# Caching is enabled by default, but can be disabled by setting
# the environment variables LOOPY_NO_CACHE or CG_NO_CACHE to a
# 'true' value.
CACHING_ENABLED = (
    not strtobool(os.environ.get("LOOPY_NO_CACHE", "false"))
    and
    not strtobool(os.environ.get("CG_NO_CACHE", "false")))


def set_caching_enabled(flag):
    """Set whether :mod:`loopy` is allowed to use disk caching for its various
    code generation stages.
    """
    global CACHING_ENABLED
    CACHING_ENABLED = flag


class CacheMode:
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
    """Returns a :class:`loopy.TranslationUnit` that changes the data layout
    of a variable (called "input") to the new layout specified by
    *new_dim_tags* from the one specified by *old_dim_tags*.
    *old_dim_tags* defaults to an all-C layout of the same rank
    as the one given by *new_dim_tags*.
    """

    from loopy.kernel.array import (
        SeparateArrayArrayDimTag,
        VectorArrayDimTag,
        parse_array_dim_tags,
    )
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
            f"0<={ind}<{shape_i}"
            for ind, shape_i in zip(indices, shape))

    set_str = "{{[{}]: {} }}".format(
                commad_indices,
                bounds
                )
    result = make_kernel(set_str,
            "output[%s] = input[%s]"
            % (commad_indices, commad_indices),
            lang_version=MOST_RECENT_LANGUAGE_VERSION,
            default_offset=auto)

    result = tag_array_axes(result, "input", old_dim_tags)
    result = tag_array_axes(result, "output", new_dim_tags)

    unrolled_tags = (SeparateArrayArrayDimTag, VectorArrayDimTag)
    for i in range(rank):
        if (isinstance(new_dim_tags[i], unrolled_tags)
                or isinstance(old_dim_tags[i], unrolled_tags)):
            result = tag_inames(result, {indices[i]: "unr"})

    return result

# }}}


# {{{ einsum

def make_einsum(spec, arg_names, **knl_creation_kwargs):
    r"""Returns a :class:`LoopKernel` for evaluating array-based
    operations using Einstein summation convention.

    :param spec: a string denoting the subscripts for
        summation as a comma-separated list of subscript labels.
        This follows the usual :func:`numpy.einsum` convention.
        Note that the explicit indicator `->` for the precise output
        form is required.
    :param arg_names: a sequence of string types denoting
        the names of the array operands.
    :param \**knl_creation_kwargs: keyword arguments for kernel creation.
        See :func:`make_kernel` for a list of acceptable keyword
        parameters.

    .. note::

        No attempt is being made to reduce the complexity
        of the resulting expression. This should be dealt with
        as part of a separate transformation.
    """
    arg_spec, out_spec = spec.split("->")
    arg_specs = arg_spec.split(",")

    out_spec = out_spec.strip()
    arg_specs = [arg_spec.strip() for arg_spec in arg_specs]

    if len(arg_names) != len(arg_specs):
        raise ValueError(
            f"Number of arg names ({arg_names}) should match the number "
            f"of arg specs: {arg_specs}. Length of arg names is {len(arg_names)}; "
            f"expecting {len(arg_specs)} arg names."
        )

    out_indices = set(out_spec)
    if len(out_indices) != len(out_spec):
        raise ValueError(
            f"Output subscripts '{out_spec}' does not contain all unique indices."
        )

    all_indices = {
        idx
        for argsp in arg_specs
        for idx in argsp} | out_indices

    sum_indices = all_indices - out_indices

    from pymbolic import var
    lhs = var("out")[tuple(var(i) for i in out_spec)]

    rhs = 1
    for arg_name, argsp in zip(arg_names, arg_specs):
        rhs = rhs * var(arg_name)[tuple(var(i) for i in argsp)]

    if sum_indices:
        rhs = Reduction("sum", tuple(var(idx) for idx in sorted(sum_indices)), rhs)

    constraints = " and ".join(
        "0 <= %s < N%s" % (idx, idx)
        for idx in all_indices
        )

    knl_creation_kwargs.setdefault("name", "einsum%dto%d_kernel" % (
            len(all_indices), len(out_indices)))
    knl_creation_kwargs.setdefault("lang_version", MOST_RECENT_LANGUAGE_VERSION)

    return make_kernel("{[%s]: %s}" % (",".join(sorted(all_indices)), constraints),
                       [Assignment(lhs, rhs)],
                       **knl_creation_kwargs)

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
        target = OpenCLTarget
    else:
        from loopy.target.pyopencl import PyOpenCLTarget
        target = PyOpenCLTarget

    set_default_target(target)


_set_up_default_target()

# }}}


# vim: foldmethod=marker

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

from loopy.diagnostic import LoopyError, warn
from pytools import Record
import islpy as isl

from pytools.persistent_dict import PersistentDict
from loopy.tools import LoopyKeyBuilder
from loopy.version import DATA_MODEL_VERSION

import logging
logger = logging.getLogger(__name__)


# {{{ implemented data info

class ImplementedDataInfo(Record):
    """
    .. attribute:: name

        The expanded name of the array. Note that, for example
        in the case of separate-array-tagged axes, multiple
        implemented arrays may correspond to one user-facing
        array.

    .. attribute:: dtype

    .. attribute:: arg_class

    .. attribute:: base_name

        The user-facing name of the underlying array.
        May be *None* for non-array arguments.

    .. attribute:: shape
    .. attribute:: strides

        Strides in multiples of ``dtype.itemsize``.

    .. attribute:: unvec_shape
    .. attribute:: unvec_strides

        Strides in multiples of ``dtype.itemsize`` that accounts for
        :class:`loopy.kernel.array.VectorArrayDimTag` in a scalar
        manner


    .. attribute:: offset_for_name
    .. attribute:: stride_for_name_and_axis

        A tuple *(name, axis)* indicating the (implementation-facing)
        name of the array and axis number for which this argument provides
        the strides.

    .. attribute:: allows_offset
    .. attribute:: is_written
    """

    def __init__(self, target, name, dtype, arg_class,
            base_name=None,
            shape=None, strides=None,
            unvec_shape=None, unvec_strides=None,
            offset_for_name=None, stride_for_name_and_axis=None,
            allows_offset=None,
            is_written=None):

        from loopy.types import LoopyType
        assert isinstance(dtype, LoopyType)

        Record.__init__(self,
                name=name,
                dtype=dtype,
                arg_class=arg_class,
                base_name=base_name,
                shape=shape,
                strides=strides,
                unvec_shape=unvec_shape,
                unvec_strides=unvec_strides,
                offset_for_name=offset_for_name,
                stride_for_name_and_axis=stride_for_name_and_axis,
                allows_offset=allows_offset,
                is_written=is_written)

# }}}


# {{{ code generation state

class Unvectorizable(Exception):
    pass


class VectorizationInfo(object):
    """
    .. attribute:: iname
    .. attribute:: length
    .. attribute:: space
    """

    def __init__(self, iname, length, space):
        self.iname = iname
        self.length = length
        self.space = space


class SeenFunction(Record):
    """
    .. attribute:: name
    .. attribute:: c_name
    .. attribute:: arg_dtypes

        a tuple of arg dtypes
    """

    def __init__(self, name, c_name, arg_dtypes):
        Record.__init__(self,
                name=name,
                c_name=c_name,
                arg_dtypes=arg_dtypes)

    def __hash__(self):
        return hash((type(self),)
                + tuple((f, getattr(self, f)) for f in type(self).fields))


class CodeGenerationState(object):
    """
    .. attribute:: kernel
    .. attribute:: implemented_data_info

        a list of :class:`ImplementedDataInfo` objects.

    .. attribute:: implemented_domain

        The entire implemented domain (as an :class:`islpy.Set`)
        i.e. all constraints that have been enforced so far.

    .. attribute:: implemented_predicates

        A :class:`frozenset` of predicates for which checks have been
        implemented.

    .. attribute:: seen_dtypes

        set of dtypes that were encountered

    .. attribute:: seen_functions

        set of :class:`SeenFunction` instances

    .. attribute:: seen_atomic_dtypes

    .. attribute:: var_subst_map

    .. attribute:: allow_complex

    .. attribute:: vectorization_info

        None or an instance of :class:`VectorizationInfo`

    .. attribute:: is_generating_device_code

    .. attribute:: gen_program_name

        None (indicating that host code is being generated)
        or the name of the device program currently being
        generated.

    .. attribute:: schedule_index_end
    """

    def __init__(self, kernel,
            implemented_data_info, implemented_domain, implemented_predicates,
            seen_dtypes, seen_functions, seen_atomic_dtypes, var_subst_map,
            allow_complex,
            vectorization_info=None, var_name_generator=None,
            is_generating_device_code=None,
            gen_program_name=None,
            schedule_index_end=None):
        self.kernel = kernel
        self.implemented_data_info = implemented_data_info
        self.implemented_domain = implemented_domain
        self.implemented_predicates = implemented_predicates
        self.seen_dtypes = seen_dtypes
        self.seen_functions = seen_functions
        self.seen_atomic_dtypes = seen_atomic_dtypes
        self.var_subst_map = var_subst_map.copy()
        self.allow_complex = allow_complex
        self.vectorization_info = vectorization_info
        self.var_name_generator = var_name_generator
        self.is_generating_device_code = is_generating_device_code
        self.gen_program_name = gen_program_name
        self.schedule_index_end = schedule_index_end

    # {{{ copy helpers

    def copy(self, kernel=None, implemented_data_info=None,
            implemented_domain=None, implemented_predicates=frozenset(),
            var_subst_map=None, vectorization_info=None,
            is_generating_device_code=None,
            gen_program_name=None,
            schedule_index_end=None):

        if kernel is None:
            kernel = self.kernel

        if implemented_data_info is None:
            implemented_data_info = self.implemented_data_info

        if vectorization_info is False:
            vectorization_info = None

        elif vectorization_info is None:
            vectorization_info = self.vectorization_info

        if is_generating_device_code is None:
            is_generating_device_code = self.is_generating_device_code

        if gen_program_name is None:
            gen_program_name = self.gen_program_name

        if schedule_index_end is None:
            schedule_index_end = self.schedule_index_end

        return CodeGenerationState(
                kernel=kernel,
                implemented_data_info=implemented_data_info,
                implemented_domain=implemented_domain or self.implemented_domain,
                implemented_predicates=(
                    implemented_predicates or self.implemented_predicates),
                seen_dtypes=self.seen_dtypes,
                seen_functions=self.seen_functions,
                seen_atomic_dtypes=self.seen_atomic_dtypes,
                var_subst_map=var_subst_map or self.var_subst_map,
                allow_complex=self.allow_complex,
                vectorization_info=vectorization_info,
                var_name_generator=self.var_name_generator,
                is_generating_device_code=is_generating_device_code,
                gen_program_name=gen_program_name,
                schedule_index_end=schedule_index_end)

    def copy_and_assign(self, name, value):
        """Make a copy of self with variable *name* fixed to *value*."""
        var_subst_map = self.var_subst_map.copy()
        var_subst_map[name] = value
        return self.copy(var_subst_map=var_subst_map)

    def copy_and_assign_many(self, assignments):
        """Make a copy of self with *assignments* included."""

        var_subst_map = self.var_subst_map.copy()
        var_subst_map.update(assignments)
        return self.copy(var_subst_map=var_subst_map)

    # }}}

    @property
    def expression_to_code_mapper(self):
        return self.ast_builder.get_expression_to_code_mapper(self)

    def intersect(self, other):
        new_impl, new_other = isl.align_two(self.implemented_domain, other)
        return self.copy(implemented_domain=new_impl & new_other)

    def fix(self, iname, aff):
        new_impl_domain = self.implemented_domain

        impl_space = self.implemented_domain.get_space()
        if iname not in impl_space.get_var_dict():
            new_impl_domain = (new_impl_domain
                    .add_dims(isl.dim_type.set, 1)
                    .set_dim_name(
                        isl.dim_type.set,
                        new_impl_domain.dim(isl.dim_type.set),
                        iname))
            impl_space = new_impl_domain.get_space()

        from loopy.isl_helpers import iname_rel_aff
        iname_plus_lb_aff = iname_rel_aff(impl_space, iname, "==", aff)

        from loopy.symbolic import pw_aff_to_expr
        cns = isl.Constraint.equality_from_aff(iname_plus_lb_aff)
        expr = pw_aff_to_expr(aff)

        new_impl_domain = new_impl_domain.add_constraint(cns)
        return self.copy_and_assign(iname, expr).copy(
                implemented_domain=new_impl_domain)

    def try_vectorized(self, what, func):
        """If *self* is in a vectorizing state (:attr:`vectorization_info` is
        not None), tries to call func (which must be a callable accepting a
        single :class:`CodeGenerationState` argument). If this fails with
        :exc:`Unvectorizable`, it unrolls the vectorized loop instead.

        *func* should return a :class:`GeneratedCode` instance.

        :returns: :class:`GeneratedCode`
        """

        if self.vectorization_info is None:
            return func(self)

        try:
            return func(self)
        except Unvectorizable as e:
            warn(self.kernel, "vectorize_failed",
                    "Vectorization of '%s' failed because '%s'"
                    % (what, e))

            return self.unvectorize(func)

    def unvectorize(self, func):
        vinf = self.vectorization_info
        result = []
        novec_self = self.copy(vectorization_info=False)

        for i in range(vinf.length):
            idx_aff = isl.Aff.zero_on_domain(vinf.space.params()) + i
            new_codegen_state = novec_self.fix(vinf.iname, idx_aff)
            generated = func(new_codegen_state)

            if isinstance(generated, list):
                result.extend(generated)
            else:
                result.append(generated)

        from loopy.codegen.result import merge_codegen_results
        return merge_codegen_results(self, result)

    @property
    def ast_builder(self):
        if self.is_generating_device_code:
            return self.kernel.target.get_device_ast_builder()
        else:
            return self.kernel.target.get_host_ast_builder()

# }}}


code_gen_cache = PersistentDict("loopy-code-gen-cache-v3-"+DATA_MODEL_VERSION,
        key_builder=LoopyKeyBuilder())


# {{{ main code generation entrypoint

def generate_code_v2(kernel):
    """
    :returns: a :class:`CodeGenerationResult`
    """

    from loopy.kernel import kernel_state
    if kernel.state == kernel_state.INITIAL:
        from loopy.preprocess import preprocess_kernel
        kernel = preprocess_kernel(kernel)

    if kernel.schedule is None:
        from loopy.schedule import get_one_scheduled_kernel
        kernel = get_one_scheduled_kernel(kernel)

    if kernel.state != kernel_state.SCHEDULED:
        raise LoopyError("cannot generate code for a kernel that has not been "
                "scheduled")

    # {{{ cache retrieval

    from loopy import CACHING_ENABLED

    if CACHING_ENABLED:
        input_kernel = kernel
        try:
            result = code_gen_cache[input_kernel]
            logger.info("%s: code generation cache hit" % kernel.name)
            return result
        except KeyError:
            pass

    # }}}

    from loopy.preprocess import infer_unknown_types
    kernel = infer_unknown_types(kernel, expect_completion=True)

    from loopy.check import pre_codegen_checks
    pre_codegen_checks(kernel)

    logger.info("%s: generate code: start" % kernel.name)

    # {{{ examine arg list

    from loopy.kernel.data import ValueArg
    from loopy.kernel.array import ArrayBase

    implemented_data_info = []

    for arg in kernel.args:
        is_written = arg.name in kernel.get_written_variables()
        if isinstance(arg, ArrayBase):
            implemented_data_info.extend(
                    arg.decl_info(
                        kernel.target,
                        is_written=is_written,
                        index_dtype=kernel.index_dtype))

        elif isinstance(arg, ValueArg):
            implemented_data_info.append(ImplementedDataInfo(
                target=kernel.target,
                name=arg.name,
                dtype=arg.dtype,
                arg_class=ValueArg,
                is_written=is_written))

        else:
            raise ValueError("argument type not understood: '%s'" % type(arg))

    allow_complex = False
    for var in kernel.args + list(six.itervalues(kernel.temporary_variables)):
        if var.dtype.involves_complex():
            allow_complex = True

    # }}}

    seen_dtypes = set()
    seen_functions = set()
    seen_atomic_dtypes = set()

    initial_implemented_domain = isl.BasicSet.from_params(kernel.assumptions)
    codegen_state = CodeGenerationState(
            kernel=kernel,
            implemented_data_info=implemented_data_info,
            implemented_domain=initial_implemented_domain,
            implemented_predicates=frozenset(),
            seen_dtypes=seen_dtypes,
            seen_functions=seen_functions,
            seen_atomic_dtypes=seen_atomic_dtypes,
            var_subst_map={},
            allow_complex=allow_complex,
            var_name_generator=kernel.get_var_name_generator(),
            is_generating_device_code=False,
            gen_program_name=(
                kernel.target.host_program_name_prefix
                + kernel.name
                + kernel.target.host_program_name_suffix),
            schedule_index_end=len(kernel.schedule))

    from loopy.codegen.result import generate_host_or_device_program
    codegen_result = generate_host_or_device_program(
            codegen_state,
            schedule_index=0)

    device_code_str = codegen_result.device_code()

    from loopy.check import check_implemented_domains
    assert check_implemented_domains(kernel, codegen_result.implemented_domains,
            device_code_str)

    # {{{ handle preambles

    for arg in kernel.args:
        seen_dtypes.add(arg.dtype)
    for tv in six.itervalues(kernel.temporary_variables):
        seen_dtypes.add(tv.dtype)

    preambles = kernel.preambles[:]

    from pytools import Record

    class PreambleInfo(Record):
        pass

    preamble_info = PreambleInfo(
            kernel=kernel,
            seen_dtypes=seen_dtypes,
            seen_functions=seen_functions,
            # a set of LoopyTypes (!)
            seen_atomic_dtypes=seen_atomic_dtypes)

    preamble_generators = (kernel.preamble_generators
            + kernel.target.get_device_ast_builder().preamble_generators())
    for prea_gen in preamble_generators:
        preambles.extend(prea_gen(preamble_info))

    codegen_result = codegen_result.copy(device_preambles=preambles)

    # }}}

    logger.info("%s: generate code: done" % kernel.name)

    if CACHING_ENABLED:
        code_gen_cache[input_kernel] = codegen_result

    return codegen_result


def generate_code(kernel, device=None):
    if device is not None:
        from warnings import warn
        warn("passing 'device' to generate_code() is deprecated",
                DeprecationWarning, stacklevel=2)

    codegen_result = generate_code_v2(kernel)

    if len(codegen_result.device_programs) > 1:
        raise LoopyError("kernel passed to generate_code yielded multiple "
                "device programs. Use generate_code_v2.")

    return codegen_result.device_code(), codegen_result.implemented_data_info

# }}}


# {{{ generate function body

def generate_body(kernel):
    codegen_result = generate_code_v2(kernel)

    if len(codegen_result.device_programs) != 1:
        raise LoopyError("generate_body cannot be used on programs "
                "that yield more than one device program")

    dev_prg, = codegen_result.device_programs

    return str(dev_prg.body_ast)

# }}}

# vim: foldmethod=marker

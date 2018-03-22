from __future__ import division, absolute_import

import re
import six

from six.moves import zip

from pytools import ImmutableRecord
from loopy.diagnostic import LoopyError

from loopy.kernel.instruction import (MultiAssignmentBase, CInstruction,
                _DataObliviousInstruction)

from loopy.symbolic import IdentityMapper, ScopedFunction


# {{{ argument descriptors

class ArgDescriptor(ImmutableRecord):
    """Base type of argument description about the variable type that is supposed to
    be encountered in a function signature.
    .. attribute:: mem_scope
    .. attribute:: shape
    .. attribute:: dim_tags
    """

    def __init__(self,
            mem_scope=None,
            shape=None,
            dim_tags=None):
        super(ArgDescriptor, self).__init__(mem_scope=mem_scope,
                shape=shape,
                dim_tags=dim_tags)


class ValueArgDescriptor(ArgDescriptor):
    def __init__(self):
        super(ValueArgDescriptor, self).__init__()

    def __str__(self):
        return "ValueArgDescriptor"

    def __repr__(self):
        return "ValueArgDescriptor"


class ArrayArgDescriptor(ArgDescriptor):
    """
    .. attribute:: mem_scope
    .. attribute:: dim_tags
    """

    def __init__(self,
            shape=None,
            mem_scope=None,
            dim_tags=None):

        # {{{ sanity checks

        assert isinstance(shape, tuple)

        # }}}

        super(ArgDescriptor, self).__init__(shape=shape,
                mem_scope=mem_scope,
                dim_tags=dim_tags)

    def copy(self, dtype=None, mem_scope=None, shape=None, dim_tags=None):
        if dtype is None:
            dtype = self.dtype

        if mem_scope is None:
            mem_scope = self.mem_scope

        if dim_tags is None:
            dim_tags = self.dim_tags

        return ArrayArgDescriptor(
                mem_scope=mem_scope,
                dim_tags=dim_tags)


# }}}


# {{{ kw_to_pos

def get_kw_pos_association(kernel):
    kw_to_pos = {}
    pos_to_kw = {}

    read_count = 0
    write_count = -1

    for arg in kernel.args:
        if arg.name in kernel.get_written_variables():
            kw_to_pos[arg.name] = write_count
            pos_to_kw[write_count] = arg.name
            write_count -= 1
        else:
            kw_to_pos[arg.name] = read_count
            pos_to_kw[read_count] = arg.name
            read_count += 1

    return kw_to_pos, pos_to_kw

# }}}


# {{{ template class


class InKernelCallable(ImmutableRecord):
    """

    .. attribute:: name

        The name of the callable which can be encountered within a kernel.

    .. attribute:: arg_id_to_dtype

        A mapping which indicates the arguments types and result types it would
        be handling. This would be set once the callable is type specialized.

    .. attribute:: arg_id_to_descr

        A mapping which gives indicates the argument shape and `dim_tags` it
        would be responsible for generating code. These parameters would be set,
        once it is shape and stride(`dim_tags`) specialized.

    .. note::

        Negative ids in the mapping attributes indicate the result arguments

    """

    def __init__(self, name, subkernel=None, arg_id_to_dtype=None,
            arg_id_to_descr=None, name_in_target=None):

        # {{{ sanity checks

        if not isinstance(name, str):
            raise LoopyError("name of an InKernelCallable should be a string")

        # }}}

        if name_in_target is not None and subkernel is not None:
            subkernel = subkernel.copy(name=name_in_target)

        super(InKernelCallable, self).__init__(name=name,
                subkernel=subkernel,
                arg_id_to_dtype=arg_id_to_dtype,
                arg_id_to_descr=arg_id_to_descr,
                name_in_target=name_in_target)

    def with_types(self, arg_id_to_dtype, target):
        """
        :arg arg_id_to_type: a mapping from argument identifiers
            (integers for positional arguments, names for keyword
            arguments) to :class:`loopy.types.LoopyType` instances.
            Unspecified/unknown types are not represented in *arg_id_to_type*.

            Return values are denoted by negative integers, with the
            first returned value identified as *-1*.

        :returns: a tuple ``(new_self, arg_id_to_type)``, where *new_self* is a
            new :class:`InKernelCallable` specialized for the given types,
            and *arg_id_to_type* is a mapping of the same form as the
            argument above, however it may have more information present.
            Any argument information exists both by its positional and
            its keyword identifier.
        """

        raise NotImplementedError()

    def with_descrs(self, arg_id_to_descr):
        """
        :arg arg_id_to_descr: a mapping from argument identifiers
            (integers for positional arguments, names for keyword
            arguments) to :class:`loopy.ArrayArgDescriptor` instances.
            Unspecified/unknown types are not represented in *arg_id_to_descr*.

            Return values are denoted by negative integers, with the
            first returned value identified as *-1*.

        :returns: a tuple ``(new_self, arg_id_to_type)``, where *new_self* is a
            new :class:`InKernelCallable` specialized for the given types,
            and *arg_id_to_descr* is a mapping of the same form as the
            argument above, however it may have more information present.
            Any argument information exists both by its positional and
            its keyword identifier.
        """

        raise NotImplementedError()

    def with_iname_tag_usage(self, unusable, concurrent_shape):
        """
        :arg unusable: a set of iname tags that may not be used in the callee.
        :arg concurrent_shape: an list of tuples ``(iname_tag, bound)`` for
            concurrent inames that are used in the calller but also available
            for mapping by the callee. *bound* is given as a
            :class:`islpy.PwAff`.

        :returns: a list of the same type as *concurrent*, potentially modified
            by increasing bounds or adding further iname tag entries.

        All iname tags not explicitly listed in *concurrent* or *unusable* are
        available for mapping by the callee.
        """

        raise NotImplementedError()

    def is_ready_for_code_gen(self):

        return (self.arg_id_to_dtype is not None and
                self.arg_id_to_descr is not None and
                self.name_in_target is not None)

    # {{{ code generation

    def generate_preambles(self, target):
        """ This would generate the target specific preamble.
        """
        raise NotImplementedError()

    def emit_call(self, expression_to_code_mapper, expression, target):

        raise NotImplementedError()

    def emit_call_insn(self, insn, target, expression_to_code_mapper):

        raise NotImplementedError()

    # }}}

    def __eq__(self, other):
        return (self.name == other.name
                and self.arg_id_to_descr == other.arg_id_to_descr
                and self.arg_id_to_dtype == other.arg_id_to_dtype
                and self.subkernel == other.subkernel)

    def __hash__(self):
        return hash((self.name, self.subkernel, self.name_in_target))


# }}}


class CallableOnScalar(InKernelCallable):

    def with_types(self, arg_id_to_dtype, target):
        if self.arg_id_to_dtype is not None:

            # specializing an already specialized function.

            for id, dtype in arg_id_to_dtype.items():
                # only checking for the ones which have been provided
                if self.arg_id_to_dtype[id] != arg_id_to_dtype[id]:
                    raise LoopyError("Overwriting a specialized"
                            " function is illegal--maybe start with new instance of"
                            " CallableScalar?")

        # {{{ attempt to specialize using scalar functions present in target

        if self.name in target.get_device_ast_builder().function_identifiers():
            new_in_knl_callable = target.get_device_ast_builder().with_types(
                    self, arg_id_to_dtype)
            if new_in_knl_callable is None:
                new_in_knl_callable = self.copy()
            return new_in_knl_callable

        # }}}

        # did not find a scalar function and function prototype does not
        # even have  subkernel registered => no match found
        raise LoopyError("Function %s not present within"
                " the %s namespace" % (self.name, target))

    def with_descrs(self, arg_id_to_descr):

        # This is a scalar call
        # need to assert that the name is in funtion indentifiers
        arg_id_to_descr[-1] = ValueArgDescriptor()
        return self.copy(arg_id_to_descr=arg_id_to_descr)

    def with_iname_tag_usage(self, unusable, concurrent_shape):

        raise NotImplementedError()

    def is_ready_for_code_gen(self):

        return (self.arg_id_to_dtype is not None and
                self.arg_id_to_descr is not None and
                self.name_in_target is not None)

    # {{{ code generation

    def generate_preambles(self, target):
        """ This would generate the target specific preamble.
        """
        raise NotImplementedError()

    def emit_call(self, expression_to_code_mapper, expression, target):

        assert self.is_ready_for_code_gen()

        # must have single assignee
        assert len(expression.parameters) == len(self.arg_id_to_dtype) - 1
        arg_dtypes = tuple(self.arg_id_to_dtype[id] for id in
                range(len(self.arg_id_to_dtype)-1))

        par_dtypes = tuple(expression_to_code_mapper.infer_type(par) for par in
                expression.parameters)

        from loopy.expression import dtype_to_type_context
        # processing the parameters with the required dtypes
        processed_parameters = tuple(
                expression_to_code_mapper.rec(par,
                    dtype_to_type_context(target, tgt_dtype),
                    tgt_dtype)
                for par, par_dtype, tgt_dtype in zip(
                    expression.parameters, par_dtypes, arg_dtypes))

        from pymbolic import var
        return var(self.name_in_target)(*processed_parameters)

    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        # TODO: Need to add support for functions like sincos(x)
        # which would give multiple outputs but takes in scalar arguments

        # FIXME: needs to get information about whether the callable has should
        # do pass by reference by all values or should return one value for
        # pass by value return.

        # For example: The code generation of `sincos` would be different for
        # C-Target and OpenCL-target.

        # Currently doing pass by value for all the assignees.

        assert self.is_ready_for_code_gen()

        from loopy.kernel.instruction import CallInstruction

        assert isinstance(insn, CallInstruction)

        parameters = insn.expression.parameters
        assignees = insn.assignees

        par_dtypes = tuple(expression_to_code_mapper.infer_type(par) for par in
                parameters)
        arg_dtypes = tuple(self.arg_id_to_dtype[i] for i, _ in
                enumerate(parameters))

        assignee_dtypes = tuple(self.arg_id_to_dtype[-i-1] for i, _ in
                enumerate(assignees))

        from loopy.expression import dtype_to_type_context
        from pymbolic.mapper.stringifier import PREC_NONE
        from pymbolic import var

        c_parameters = [
                expression_to_code_mapper(par, PREC_NONE,
                    dtype_to_type_context(self.target, tgt_dtype),
                    tgt_dtype).expr
                for par, par_dtype, tgt_dtype in zip(
                    parameters, par_dtypes, arg_dtypes)]

        for i, (a, tgt_dtype) in enumerate(zip(assignees, assignee_dtypes)):
            if tgt_dtype != expression_to_code_mapper.infer_type(a):
                raise LoopyError("Type Mismach in funciton %s. Expected: %s"
                        "Got: %s" % (self.name, tgt_dtype,
                            expression_to_code_mapper.infer_type(a)))
            c_parameters.append(
                        var("&")(
                            expression_to_code_mapper(a, PREC_NONE,
                                dtype_to_type_context(self.target, tgt_dtype),
                                tgt_dtype).expr))

        from pymbolic import var
        return var(self.name_in_target)(*c_parameters)

        raise NotImplementedError("emit_call_insn only applies for"
                " CallableKernels")

    # }}}


class CallableKernel(InKernelCallable):

    def with_types(self, arg_id_to_dtype, target):

        kw_to_pos, pos_to_kw = get_kw_pos_association(self.subkernel)

        new_args = []
        for arg in self.subkernel.args:
            kw = arg.name
            if kw in arg_id_to_dtype:
                # id exists as kw
                new_args.append(arg.copy(dtype=arg_id_to_dtype[kw]))
            elif kw_to_pos[kw] in arg_id_to_dtype:
                # id exists as positional argument
                new_args.append(arg.copy(
                    dtype=arg_id_to_dtype[kw_to_pos[kw]]))
            else:
                if kw in self.subkernel.get_read_variables():
                    # need to know the type of the input arguments for type
                    # inference
                    raise LoopyError("Type of %s variable not supplied to the"
                            " subkernel, which is needed for type"
                            " inference." % kw)
                new_args.append(arg)

        from loopy.type_inference import infer_unknown_types
        pre_specialized_subkernel = self.subkernel.copy(
                args=new_args)

        # inferring the types of the written variables based on the knowledge
        # of the types of the arguments supplied
        specialized_kernel = infer_unknown_types(pre_specialized_subkernel,
                expect_completion=True)
        new_arg_id_to_dtype = {}
        read_count = 0
        write_count = -1
        for arg in specialized_kernel.args:
            new_arg_id_to_dtype[arg.name] = arg.dtype
            if arg.name in specialized_kernel.get_written_variables():
                new_arg_id_to_dtype[write_count] = arg.dtype
                write_count -= 1
            else:
                new_arg_id_to_dtype[read_count] = arg.dtype
                read_count += 1

        # Returning the kernel call with specialized subkernel and the corresponding
        # new arg_id_to_dtype
        return self.copy(subkernel=specialized_kernel,
                arg_id_to_dtype=new_arg_id_to_dtype)

    def with_descrs(self, arg_id_to_descr):

        # tuning the subkernel so that we have the the matching shapes and
        # dim_tags.
        # FIXME: Although We receive input if the argument is
        # `local/global`. We do not use it to set the subkernel function
        # signature. Need to do it, so that we can handle teporary inputs
        # in the array call.

        # Collecting the parameters
        new_args = self.subkernel.args.copy()
        kw_to_pos, pos_to_kw = get_kw_pos_association(self.subkernel)

        for id, descr in arg_id_to_descr.items():
            if isinstance(id, str):
                id = kw_to_pos[id]
            assert isinstance(id, int)
            new_args[id] = new_args[id].copy(shape=descr.shape,
                    dim_tags=descr.dim_tags)

        descriptor_specialized_knl = self.subkernel.copy(args=new_args)

        return self.copy(subkernel=descriptor_specialized_knl,
                arg_id_to_descr=arg_id_to_descr)

    def with_iname_tag_usage(self, unusable, concurrent_shape):

        raise NotImplementedError()

    def is_ready_for_code_gen(self):

        return (self.arg_id_to_dtype is not None and
                self.arg_id_to_descr is not None and
                self.name_in_target is not None)

    # {{{ code generation

    def generate_preambles(self, target):
        """ This would generate the target specific preamble.
        """
        raise NotImplementedError()

    def emit_call(self, expression_to_code_mapper, expression, target):

        raise NotImplementedError("emit_call only works on scalar operations")

    def emit_call_insn(self, insn, target, expression_to_code_mapper):

        assert self.is_ready_for_code_gen()

        from loopy.kernel.instruction import CallInstruction
        from pymbolic.primitives import CallWithKwargs

        assert isinstance(insn, CallInstruction)

        parameters = insn.expression.parameters
        kw_parameters = {}
        if isinstance(insn.expression, CallWithKwargs):
            kw_parameters = insn.expression.kw_parameters

        assignees = insn.assignees

        parameters = list(parameters)
        par_dtypes = [self.arg_id_to_dtype[i] for i, _ in enumerate(parameters)]
        kw_to_pos, pos_to_kw = get_kw_pos_association(self.subkernel)
        for i in range(len(parameters), len(parameters)+len(kw_parameters)):
            parameters.append(kw_parameters[pos_to_kw[i]])
            par_dtypes.append(self.arg_id_to_dtype[pos_to_kw[i]])

        # TODO: currently no suppport for assignee keywords.
        parameters = parameters + list(assignees)
        par_dtypes = par_dtypes + [self.arg_id_to_dtype[-i-1] for i, _ in
                enumerate(assignees)]

        # Note that we are not going to do any type casting in array calls.
        from loopy.expression import dtype_to_type_context
        from pymbolic.mapper.stringifier import PREC_NONE
        c_parameters = [
                expression_to_code_mapper(par, PREC_NONE,
                    dtype_to_type_context(target, par_dtype),
                    par_dtype).expr
                for par, par_dtype in zip(
                    parameters, par_dtypes)]

        from pymbolic import var
        return var(self.name_in_target)(*c_parameters)

    # }}}

    def __eq__(self, other):
        return (self.name == other.name
                and self.arg_id_to_descr == other.arg_id_to_descr
                and self.arg_id_to_dtype == other.arg_id_to_dtype
                and self.subkernel == other.subkernel)

    def __hash__(self):
        return hash((self.name, self.subkernel, self.name_in_target))


# {{{ new pymbolic calls to scoped functions

def next_indexed_name(name):
    func_name = re.compile(r"^(?P<alpha>\S+?)_(?P<num>\d+?)$")

    match = func_name.match(name)

    if match is None:
        if name[-1] == '_':
            return "{old_name}0".format(old_name=name)
        else:
            return "{old_name}_0".format(old_name=name)

    return "{alpha}_{num}".format(alpha=match.group('alpha'),
            num=int(match.group('num'))+1)


class FunctionScopeChanger(IdentityMapper):
    #TODO: Make it sophisticated as in I don't like the if-else systems. Needs
    # something else.
    def __init__(self, new_names):
        self.new_names = new_names
        self.new_names_set = frozenset(new_names.values())

    def map_call(self, expr):
        if expr in self.new_names:
            return type(expr)(
                    ScopedFunction(self.new_names[expr]),
                    tuple(self.rec(child)
                        for child in expr.parameters))
        else:
            return IdentityMapper.map_call(self, expr)

    def map_call_with_kwargs(self, expr):
        if expr in self.new_names:
            return type(expr)(
                ScopedFunction(self.new_names[expr]),
                tuple(self.rec(child)
                    for child in expr.parameters),
                dict(
                    (key, self.rec(val))
                    for key, val in six.iteritems(expr.kw_parameters))
                    )
        else:
            return IdentityMapper.map_call_with_kwargs(self, expr)


def register_pymbolic_calls_to_knl_callables(kernel,
        pymbolic_calls_to_knl_callables):
    """ Takes in a mapping :arg:`pymbolic_calls_to_knl_callables` and returns a
    new kernel which includes an association with the given pymbolic calls to
    instances of :class:`InKernelCallable`
    """

    scoped_names_to_functions = kernel.scoped_functions.copy()

    # A dict containing the new scoped functions to the names which have been
    # assigned to them
    scoped_functions_to_names = {}

    # A dict containing the new name that need to be assigned to the
    # corresponding pymbolic call
    pymbolic_calls_to_new_names = {}

    for pymbolic_call, in_knl_callable in pymbolic_calls_to_knl_callables.items():
        # checking if such a in-kernel callable already exists.
        if in_knl_callable not in scoped_functions_to_names:
            # No matching in_knl_callable found => make a new one with a new
            # name.

            unique_name = next_indexed_name(pymbolic_call.function.name)
            while unique_name in scoped_names_to_functions:
                # keep on finding new names till one a unique one is found.
                unique_name = next_indexed_name(unique_name)

            # book-keeping of the functions and names mappings for later use
            if in_knl_callable.subkernel is not None:
                # for array calls the name in the target is the name of the
                # scoped funciton
                in_knl_callable = in_knl_callable.copy(
                        name_in_target=unique_name)
            scoped_names_to_functions[unique_name] = in_knl_callable
            scoped_functions_to_names[in_knl_callable] = unique_name

        pymbolic_calls_to_new_names[pymbolic_call] = (
                scoped_functions_to_names[in_knl_callable])

    # Using the data populated in pymbolic_calls_to_new_names to change the
    # names of the scoped functions of all the calls in the kernel.
    new_insns = []
    scope_changer = FunctionScopeChanger(pymbolic_calls_to_new_names)
    for insn in kernel.instructions:
        if isinstance(insn, (MultiAssignmentBase, CInstruction)):
            expr = scope_changer(insn.expression)
            new_insns.append(insn.copy(expression=expr))
        elif isinstance(insn, _DataObliviousInstruction):
            new_insns.append(insn)
        else:
            raise NotImplementedError("Type Inference Specialization not"
                    "implemented for %s instruciton" % type(insn))
    return kernel.copy(scoped_functions=scoped_names_to_functions,
            instructions=new_insns)

# }}}

# vim: foldmethod=marker

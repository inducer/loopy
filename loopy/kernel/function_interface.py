from __future__ import division, absolute_import

import numpy as np

from pytools import ImmutableRecord
from loopy.diagnostic import LoopyError

from loopy.types import NumpyType


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
        super(ArgDescriptor).__init__(self,
                mem_scope=mem_scope,
                shape=shape,
                dim_tags=dim_tags)


class ValueArgDescriptor(ArgDescriptor):
    """
    """
    def __init__(self):
        super(ValueArgDescriptor, self).__init__(self)


class ArrayArgDescriptor(ArgDescriptor):
    """
    .. attribute:: mem_scope
    .. attribute:: dim_tags
    """

    def __init__(self,
            mem_scope=None,
            dim_tags=None):
        super(ArgDescriptor, self).__init__(self,
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


# {{{ in kernel callable

class InKernelCallable(ImmutableRecord):
    """

    .. attribute:: name

        The name of the callable which can be encountered within a kernel.

    .. note::

        Negative ids in the mapping attributes indicate the result arguments

    """

    def __init__(self, name=None):

        # {{{ sanity checks

        if not isinstance(name, str):
            raise LoopyError("name of a InKernelCallable should be a string")

        # }}}

        self.name = name

        super(InKernelCallable, self).__init__(name=name)

    def copy(self, name=None):
        if name is None:
            name = self.name

        return InKernelCallable(name=name)

    def with_types(self, arg_id_to_dtype):
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

    def is_arg_written(self, arg_id):
        """
        :arg arg_id: (keyword) name or position
        """

        raise NotImplementedError()

    def is_ready_for_code_gen(self):

        raise NotImplementedError()

    # {{{ code generation

    def generate_preambles(self, target):
        """ This would generate the target specific preamble.
        """
        raise NotImplementedError()

    def get_target_specific_name(self, target):

        raise NotImplementedError()

    def emit_call(self, target):

        raise NotImplementedError()

    # }}}

    def __eq__(self, other):
        return (self.name == other.name
                and self.arg_id_to_descr == other.arg_id_to_descr
                and self.arg_id_to_keyword == other.arg_id_to_keyword)

    def __hash__(self):
        return hash((self.name, ))

# }}}


# {{{ generic callable class


class CommonReturnTypeCallable(InKernelCallable):
    """ A class of generic functions which have the following properties:
        - Single return value
        - Return type of the callable is a common dtype to all the input arguments
          to the callable

    .. attribute:: name

        The name of the function as would be encountered in loopy.

    ..attribute:: specialized_dtype

        The dtype for which the function has been setup to generate code and
        premables. For example, the function `sin` can be specialized to either one
        of the following `float sin(float x)` or `double sin(double x)`. This is not
        usually expected to be an input as this removed the generality of the
        callable.

    ..attribute:: kinds_allowed

        The extent upto which the function can be generalized upto. For example
        `sin(x)` cannot have complex types as its specialized type.

    ..attribute:: arity

        The number of inputs that are to be given to the function

    """

    def __init__(self, name=None, specialized_dtype=None, kinds_allowed=None,
            arity=None):

        super(CommonReturnTypeCallable, self).__init__(name=name)

        self.specialized_dtype = specialized_dtype
        self.kinds_allowed = kinds_allowed
        self.arity = arity

    def copy(self, specialized_dtype=None):
        if specialized_dtype is None:
            specialized_dtype = self.specialized_dtype

        return type(self)(self.name, specialized_dtype,
                self.kinds_allowed, self.arity)

    def with_types(self, arg_id_to_dtype):

        specialized_dtype = np.find_common_type([], [dtype.numpy_dtype
            for id, dtype in arg_id_to_dtype.items() if id >= 0])

        if self.specialized_dtype is not None and (specialized_dtype !=
                self.specialized_dtype):
            from loopy.warnings import warn
            warn("Trying to change the type of the already set function."
                    "-- maybe use a different class instance?")

        new_arg_id_to_dtype = arg_id_to_dtype.copy()
        # checking the compliance of the arg_id_to_dtype

        if -1 not in arg_id_to_dtype:
            # return type was not know earlier, now setting it to the common type
            new_arg_id_to_dtype[-1] = NumpyType(specialized_dtype)

        if self.arity+1 == len(new_arg_id_to_dtype) and (specialized_dtype.kind in
                self.kinds_allowed):
            # the function signature matched with the current instance.
            # returning the function and the new_arg_id_to_dtype
            for i in range(self.arity):
                new_arg_id_to_dtype[i] = NumpyType(specialized_dtype)

            return (self.copy(specialized_dtype=specialized_dtype),
                    new_arg_id_to_dtype)

        return None

    def is_ready_for_code_gen(self):
        return self.specilized_dtype is not None

    def get_target_specific_name(self, target):
        raise NotImplementedError()

    def get_preamble(self, target):
        raise NotImplementedError()

# }}}

# {{{ specific type callable class


class SpecificReturnTypeCallable(InKernelCallable):
    """ A super class for the funcitons which cannot be listed as generic
    functions. These types of Callables support explicity mentioning of the
    arguments and result dtypes.

    .. attribute:: name

        The name of the function as would be encountered in loopy.

    .. attribute:: arg_id_to_dtype

        The dtype pattern of the arguments which is supposed to be used for checking
        the applicability of this function in a given scenario.
    """

    def __init__(self, name=None, arg_id_to_dtype=None):

        super(SpecificReturnTypeCallable, self).__init__(name=name)

        if arg_id_to_dtype is None:
            LoopyError("The function signature is incomplete without the"
                    "`arg_id_to_dtype`")
        self.arg_id_to_dtype = arg_id_to_dtype

    def with_types(self, arg_id_to_dtype):

        # Checking the number of inputs
        if len([id for id in arg_id_to_dtype if id >= 0]) != len(
                [id for id in self.arg_id_to_dtype if id >= 0]):
            # the number of input arguments do not match
            return None

        # Checking the input dtypes
        for id, dtype in arg_id_to_dtype.items():
            if id in self.arg_id_to_dtype and self.arg_id_to_dtype[id] == dtype:
                # dtype matched with the one given in the input
                pass
            else:
                # did not match with  the function signature and hence returning
                # None
                return None

        # Setting the output if not present
        new_arg_id_to_dtype = arg_id_to_dtype.copy()
        for id, dtype in self.arg_id_to_dtype:
            if id < 0:
                # outputs
                if id in new_arg_id_to_dtype and new_arg_id_to_dtype[id] != dtype:
                    # the output dtype had been supplied but did not match with the
                    # one in the function signature
                    return None

                new_arg_id_to_dtype[id] = dtype

        # Finally returning the types
        return self.copy(), new_arg_id_to_dtype

    def is_ready_for_code_gen(self):
        # everything about the function is determined at the constructor itself,
        # hence always redy for codegen
        return True

    def get_target_specific_name(self, target):
        # defaults to the name of the function in Loopy. May change this specific to
        # a target by inheriting this class and overriding this function.
        return self.name

    def get_preamble(self, target):
        return ""

# }}}

# {{{ callable kernel


class CallableKernel(InKernelCallable):
    """

    ..attribute:: name

        This would be the name by which the function would be called in the loopy
        kernel.

    .. attribute:: subkernel

        The subkernel associated with the call.

    """

    # {{{ constructor

    def __init__(self, name=None, subkernel=None):

        super(CallableKernel, self).__init__(name=name)

        if not name == subkernel.name:
            subkernel = subkernel.copy(name=name)

        self.subkernel = subkernel

    # }}}

    # {{{ copy

    def copy(self, name=None, subkernel=None):
        if name is None:
            name = self.name

        if subkernel is None:
            subkernel = self.subkernel

        return self.__class__(name=name,
                subkernel=subkernel)

    # }}}

    # {{{ with_types

    def with_types(self, arg_id_to_dtype):

        # {{{ sanity checks for arg_id_to_dtype

        for id in arg_id_to_dtype:
            if not isinstance(id, str):
                raise LoopyError("For Callable kernels the input should be all given"
                        "as KWargs")

        # }}}

        # Checking the input dtypes
        for id, arg in self.subkernel.arg_dict.items():
            if id in self.subkernel.read_varibles():

                # because we need the type of the parameters from the main kernel. It
                # is necessary that we know the types from there. Hence asserting
                # this condition
                assert id in arg_id_to_dtype

        new_arg_dict = {}
        for id, dtype in arg_id_to_dtype.items():
            # Making the type of the new arg according to the arg which has been
            # called in the function.
            new_arg_dict[id] = self.subkernel.arg_dict[id].copy(dtype=dtype)

        # Merging the 2 dictionaries so that to even incorporate the variables that
        # were not mentioned in arg_id_to_dtype.
        new_arg_dict = {**self.subkernel.arg_dict, **new_arg_dict}

        # Preprocessing the kernel so that we can get the types of the other
        # variables that are involved in the args
        from loopy.type_inference import infer_unknown_types
        pre_specialized_subkernel = self.subkernel.copy(
                args=list(new_arg_dict.values))

        # inferring the types of the written variables based on the knowledge of the
        # types of the arguments supplied
        specialized_kernel = infer_unknown_types(pre_specialized_subkernel,
                expect_completion=True)

        new_arg_id_to_dtype = {}
        for id, arg in specialized_kernel.arg_dict:
            new_arg_id_to_dtype[id] = arg.dtype

        # Returning the kernel call with specialized subkernel and the corresponding
        # new arg_id_to_dtype
        return self.copy(subkernel=specialized_kernel), specialized_kernel.arg_dict

    # }}}

    # {{{ with_descriptors

    def with_descriptors(self, arg_id_to_descr):
        for id, arg_descr in arg_id_to_descr.items():
            # The dimensions don't match => reject it
            if len(arg_descr.dim_tags) != len(self.subkernel.arg_dict[id].shape):
                raise LoopyError("The number of dimensions do not match between the"
                        "caller kernel and callee kernel for the variable name %s in"
                        "the callee kernel" % id)

        new_args = []
        for arg in self.subkernel.args:
            if arg.name in arg_id_to_descr:
                new_args.copy(arg.copy(dim_tags=arg_id_to_descr[arg.name]))
                pass
            else:
                new_args.append(arg.copy())

        specialized_kernel = self.subkernel.copy(args=new_args)

        new_arg_id_to_descr = {}

        for id, arg in specialized_kernel.arg_dict.items():
            new_arg_id_to_descr[id] = ArrayArgDescriptor(arg.dim_tags, "GLOBAL")

        return self.copy(subkernel=specialized_kernel), new_arg_id_to_descr

    # }}}

    # {{{ get_target_specific_name

    def get_target_specific_name(self, target):
        return self.subkernel.name

    # }}}

    # {{{ get preamble

    def get_preamble(self, target):
        return ""

    # }}}

# }}}

# vim: foldmethod=marker

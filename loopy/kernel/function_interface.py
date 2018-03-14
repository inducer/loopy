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


# {{{ c with types

def c_with_types(name, arg_id_to_dtype):

    # Specializing the type of the math function once they agree upon the
    # function signature.

    if name in ["abs", "acos", "asin", "atan", "cos", "cosh", "sin", "sinh",
            "tanh", "exp", "log", "log10", "sqrt", "ceil", "floor", "tan"]:
        for id, dtype in arg_id_to_dtype.items():
            if not -1 <= id <= 0:
                raise LoopyError("%s can take only one argument." % name)

        dtype = arg_id_to_dtype[0].numpy_dtype

        if dtype.kind == 'f':
            # generic type resolve we can go ahead and specialize
            pass
        elif dtype.kind in ['u', 'i']:
            # int and unsigned are casted into float32
            dtype = np.float32
        else:
            raise LoopyError("%s function cannot take arguments of the type %s"
                    % (name, dtype))

        # Done specializing. Returning the intended arg_id_to_dtype
        dtype = NumpyType(dtype)
        return {-1: dtype, 0: dtype}

    # binary functions
    elif name in ["max", "min"]:
        for id, dtype in arg_id_to_dtype.items():
            if not -1 <= id <= 1:
                raise LoopyError("%s can take only two arguments." % name)

        # finding the common type for all the dtypes involved
        dtype = np.find_common_type(
            [], [dtype.numpy_dtype for dtype in arg_id_to_dtype])

        if dtype.kind == 'f':
            # generic type resolve we can go ahead and specialize
            pass
        elif dtype.kind in ['u', 'i']:
            # int and unsigned are implicitly casted into float32
            dtype = np.float32
        else:
            raise LoopyError("%s function cannot take arguments of the type %s"
                    % (name, dtype))

        # Specialized into one of the known types
        return {-1: NumpyType(dtype), 0: arg_id_to_dtype[0], 1: arg_id_to_dtype[1]}

    else:
        # could not specialize the function within the C namespace
        # this would help when checking for OpenCL/CUDA function which are not
        # present in C
        return None

# }}}


# {{{ opencl with_types

def opencl_with_types(name, arg_id_to_dtype):
    new_arg_id_to_dtype = c_with_types(name, arg_id_to_dtype)
    if new_arg_id_to_dtype is None:
        # could not locate the function within C's namespace. Searching in
        # OpenCL specific namespace

        # FIXME: Need to add these functions over here
        new_arg_id_to_dtype = None

    return new_arg_id_to_dtype

# }}}


# {{{ pyopencl with_types

def pyopencl_with_types(name, arg_id_to_dtype):
    new_arg_id_to_dtype = opencl_with_types(name, arg_id_to_dtype)
    if new_arg_id_to_dtype is None:
        # could not locate the function within C's namespace. Searching in
        # PyOpenCL specific namespace

        # FIXME: Need to add these functions over here
        new_arg_id_to_dtype = None

    return new_arg_id_to_dtype

# }}}


# {{{ cuda with_types

def cuda_with_types(name, arg_id_to_dtype):
    new_arg_id_to_dtype = c_with_types(name, arg_id_to_dtype)
    if new_arg_id_to_dtype is None:
        # could not locate the function within C's namespace. Searching in
        # CUDA specific namespace

        # FIXME: Need to add these extra functions over here
        new_arg_id_to_dtype = None

    return new_arg_id_to_dtype

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
            arg_id_to_descr=None):

        # {{{ sanity checks

        if not isinstance(name, str):
            raise LoopyError("name of a InKernelCallable should be a string")

        # }}}

        super(InKernelCallable, self).__init__(name=name,
                subkernel=subkernel,
                arg_id_to_dtype=arg_id_to_dtype,
                arg_id_to_descr=arg_id_to_descr)

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

        if self.arg_id_to_dtype:
            # trying to specialize an already specialized function.

            if self.arg_id_to_dtype == arg_id_to_dtype:
                return self.copy()
            else:
                raise LoopyError("Overwriting a specialized function--maybe"
                        " start with new instance of InKernelCallable?")

        # {{{ attempt to specialize using scalar functions

        from loopy.library.function import default_function_identifiers
        if self.name in default_function_identifiers():
            ...
        elif self.name in target.get_device_ast_builder().function_identifiers():
            from loopy.target.c import CTarget
            from loopy.target.opencl import OpenCLTarget
            from loopy.target.pyopencl import PyOpenCLTarget
            from loopy.target.cuda import CudaTarget

            if isinstance(target, CTarget):
                new_arg_id_to_dtype = c_with_types(self.name, arg_id_to_dtype)

            elif isinstance(target, OpenCLTarget):
                new_arg_id_to_dtype = opencl_with_types(self.name, arg_id_to_dtype)

            elif isinstance(target, PyOpenCLTarget):
                new_arg_id_to_dtype = pyopencl_with_types(self.name, arg_id_to_dtype)

            elif isinstance(target, CudaTarget):
                new_arg_id_to_dtype = cuda_with_types(self.name, arg_id_to_dtype)

            else:
                raise NotImplementedError("InKernelCallable.with_types() for"
                        " %s target" % target)

        # }}}

        if new_arg_id_to_dtype is not None:
            # got our speciliazed function
            return self.copy(arg_id_to_dtype=new_arg_id_to_dtype)

        if self.subkernel is None:
            # did not find a scalar function and function prototype does not
            # even have  subkernel registered => no match found
            raise LoopyError("Function %s not present within"
                    " the %s namespace" % (self.name, target))

        # {{{ attempt to specialization with array functions

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
                if kw in self.subkernel.read_variables():
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

        # }}}

        # Returning the kernel call with specialized subkernel and the corresponding
        # new arg_id_to_dtype
        return self.copy(subkernel=specialized_kernel,
                arg_id_to_dtype=new_arg_id_to_dtype)

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
                and self.arg_id_to_dtype == other.arg_id_to_keyword)

    def __hash__(self):
        return hash((self.name, self.subkernel))

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

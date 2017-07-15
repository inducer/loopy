from __future__ import division, with_statement, absolute_import

__copyright__ = "Copyright (C) 2012-16 Andreas Kloeckner"

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
import numpy as np
from pytools import ImmutableRecord, memoize_method
from loopy.diagnostic import LoopyError

import logging
logger = logging.getLogger(__name__)

from pytools.persistent_dict import PersistentDict
from loopy.tools import LoopyKeyBuilder
from loopy.version import DATA_MODEL_VERSION


# {{{ object array argument packing

class _PackingInfo(ImmutableRecord):
    """
    .. attribute:: name
    .. attribute:: sep_shape

    .. attribute:: subscripts_and_names

        A list of type ``[(index, unpacked_name), ...]``.
    """


class SeparateArrayPackingController(object):
    """For argument arrays with axes tagged to be implemented as separate
    arrays, this class provides preprocessing of the incoming arguments so that
    all sub-arrays may be passed in one object array (under the original,
    un-split argument name) and are unpacked into separate arrays before being
    passed to the kernel.

    It also repacks outgoing arrays of this type back into an object array.
    """

    def __init__(self, kernel):
        # map from arg name
        self.packing_info = {}

        from loopy.kernel.array import ArrayBase
        for arg in kernel.args:
            if not isinstance(arg, ArrayBase):
                continue

            if arg.shape is None or arg.dim_tags is None:
                continue

            subscripts_and_names = arg.subscripts_and_names()

            if subscripts_and_names is None:
                continue

            self.packing_info[arg.name] = _PackingInfo(
                    name=arg.name,
                    sep_shape=arg.sep_shape(),
                    subscripts_and_names=subscripts_and_names,
                    is_written=arg.name in kernel.get_written_variables())

    def unpack(self, kernel_kwargs):
        if not self.packing_info:
            return kernel_kwargs

        kernel_kwargs = kernel_kwargs.copy()

        for packing_info in six.itervalues(self.packing_info):
            arg_name = packing_info.name
            if packing_info.name in kernel_kwargs:
                arg = kernel_kwargs[arg_name]
                for index, unpacked_name in packing_info.subscripts_and_names:
                    assert unpacked_name not in kernel_kwargs
                    kernel_kwargs[unpacked_name] = arg[index]
                del kernel_kwargs[arg_name]

        return kernel_kwargs

    def pack(self, outputs):
        if not self.packing_info:
            return outputs

        for packing_info in six.itervalues(self.packing_info):
            if not packing_info.is_written:
                continue

            result = outputs[packing_info.name] = \
                    np.zeros(packing_info.sep_shape, dtype=np.object)

            for index, unpacked_name in packing_info.subscripts_and_names:
                result[index] = outputs.pop(unpacked_name)

        return outputs

# }}}


# {{{ KernelExecutorBase

typed_and_scheduled_cache = PersistentDict(
        "loopy-typed-and-scheduled-cache-v1-"+DATA_MODEL_VERSION,
        key_builder=LoopyKeyBuilder())


class KernelExecutorBase(object):
    """An object connecting a kernel to a :class:`pyopencl.Context`
    for execution.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, kernel):
        """
        :arg kernel: a loopy.LoopKernel
        """

        self.kernel = kernel

        self.packing_controller = SeparateArrayPackingController(kernel)

        self.output_names = tuple(arg.name for arg in self.kernel.args
                if arg.name in self.kernel.get_written_variables())

        self.has_runtime_typed_args = any(
                arg.dtype is None
                for arg in kernel.args)

    def get_typed_and_scheduled_kernel_uncached(self, arg_to_dtype_set):
        from loopy.kernel.tools import add_dtypes

        kernel = self.kernel

        if arg_to_dtype_set:
            var_to_dtype = {}
            for var, dtype in arg_to_dtype_set:
                try:
                    dest_name = kernel.impl_arg_to_arg[var].name
                except KeyError:
                    dest_name = var

                try:
                    var_to_dtype[dest_name] = dtype
                except KeyError:
                    raise LoopyError("cannot set type for '%s': "
                            "no known variable/argument with that name"
                            % var)

            kernel = add_dtypes(kernel, var_to_dtype)

            from loopy.type_inference import infer_unknown_types
            kernel = infer_unknown_types(kernel, expect_completion=True)

        if kernel.schedule is None:
            from loopy.preprocess import preprocess_kernel
            kernel = preprocess_kernel(kernel)

            from loopy.schedule import get_one_scheduled_kernel
            kernel = get_one_scheduled_kernel(kernel)

        return kernel

    @memoize_method
    def get_typed_and_scheduled_kernel(self, arg_to_dtype_set):
        from loopy import CACHING_ENABLED

        from loopy.preprocess import prepare_for_caching
        # prepare_for_caching() gets run by preprocess, but the kernel at this
        # stage is not guaranteed to be preprocessed.
        cacheable_kernel = prepare_for_caching(self.kernel)
        cache_key = (type(self).__name__, cacheable_kernel, arg_to_dtype_set)

        if CACHING_ENABLED:
            try:
                return typed_and_scheduled_cache[cache_key]
            except KeyError:
                pass

        logger.debug("%s: typed-and-scheduled cache miss" % self.kernel.name)

        kernel = self.get_typed_and_scheduled_kernel_uncached(arg_to_dtype_set)

        if CACHING_ENABLED:
            typed_and_scheduled_cache[cache_key] = kernel

        return kernel

    def arg_to_dtype_set(self, kwargs):
        if not self.has_runtime_typed_args:
            return None

        from loopy.types import NumpyType
        target = self.kernel.target

        impl_arg_to_arg = self.kernel.impl_arg_to_arg
        arg_to_dtype = {}
        for arg_name, val in six.iteritems(kwargs):
            arg = impl_arg_to_arg.get(arg_name, None)

            if arg is None:
                # offsets, strides and such
                continue

            if arg.dtype is None and val is not None:
                try:
                    dtype = val.dtype
                except AttributeError:
                    pass
                else:
                    arg_to_dtype[arg_name] = NumpyType(dtype, target)

        return frozenset(six.iteritems(arg_to_dtype))

# }}}

# vim: foldmethod=marker

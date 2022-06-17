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

from typing import TYPE_CHECKING, Tuple, Optional
from dataclasses import dataclass
from warnings import warn

import numpy as np

import loopy as lp
from loopy.kernel.array import get_strides

from loopy.diagnostic import LoopyError, AutomaticTestFailure

if TYPE_CHECKING:
    import pyopencl.array as cla


AUTO_TEST_SKIP_RUN = False

import logging
logger = logging.getLogger(__name__)


def is_dtype_supported(dtype):
    # Work around https://github.com/numpy/numpy/issues/4317
    return dtype.kind in "biufc"


def evaluate_shape(shape, context):
    from pymbolic import evaluate

    result = []
    for saxis in shape:
        if saxis is None:
            result.append(saxis)
        else:
            result.append(evaluate(saxis, context))

    return tuple(result)


# {{{ create random argument arrays for testing

def fill_rand(ary):
    from pyopencl.clrandom import fill_rand
    if ary.dtype.kind == "c":
        real_dtype = ary.dtype.type(0).real.dtype
        real_ary = ary.view(real_dtype)

        fill_rand(real_ary)
    else:
        fill_rand(ary)


@dataclass
class TestArgInfo:
    name: str
    ref_array: "cla.Array"
    ref_storage_array: "cla.Array"

    ref_pre_run_array: "cla.Array"
    ref_pre_run_storage_array: "cla.Array"

    ref_shape: Tuple[int, ...]
    ref_strides: Tuple[int, ...]
    ref_alloc_size: int
    ref_numpy_strides: Tuple[int, ...]
    needs_checking: bool

    # The attributes below are being modified in make_args, hence this dataclass
    # cannot be frozen.
    test_storage_array: Optional["cla.Array"] = None
    test_array: Optional["cla.Array"] = None
    test_shape: Optional[Tuple[int, ...]] = None
    test_strides: Optional[Tuple[int, ...]] = None
    test_numpy_strides: Optional[Tuple[int, ...]] = None
    test_alloc_size: Optional[Tuple[int, ...]] = None


# {{{ "reference" arguments

def make_ref_args(kernel, queue, parameters):
    import pyopencl as cl
    import pyopencl.array as cl_array

    from loopy.kernel.data import ValueArg, ArrayArg, ImageArg, \
            TemporaryVariable, ConstantArg

    from pymbolic import evaluate

    ref_args = {}
    ref_arg_data = []

    for arg in kernel.args:
        if isinstance(arg, ValueArg):
            arg_value = parameters[arg.name]

            try:
                argv_dtype = arg_value.dtype
            except AttributeError:
                argv_dtype = None

            if argv_dtype != arg.dtype:
                arg_value = arg.dtype.numpy_dtype.type(arg_value)

            ref_args[arg.name] = arg_value

            ref_arg_data.append(None)

        elif isinstance(arg, (ArrayArg, ImageArg, ConstantArg)):
            if arg.shape is None or any(saxis is None for saxis in arg.shape):
                raise LoopyError("array '%s' needs known shape to use automatic "
                        "testing" % arg.name)

            shape = evaluate_shape(arg.shape, parameters)
            dtype = arg.dtype

            is_output = arg.is_output

            if isinstance(arg, ImageArg):
                storage_array = ary = cl_array.empty(
                        queue, shape, dtype, order="C")
                numpy_strides = None
                alloc_size = None
                strides = None
            else:
                strides = evaluate(get_strides(arg), parameters)

                alloc_size = sum(astrd*(alen-1) if astrd != 0 else alen-1
                        for alen, astrd in zip(shape, strides)) + 1

                if dtype is None:
                    raise LoopyError("dtype for argument '%s' is not yet "
                            "known. Perhaps you want to use "
                            "loopy.add_dtypes "
                            "or loopy.infer_argument_dtypes?"
                            % arg.name)

                itemsize = dtype.itemsize
                numpy_strides = [itemsize*s for s in strides]

                storage_array = cl_array.empty(queue, alloc_size, dtype)

            if is_output and isinstance(arg, ImageArg):
                raise LoopyError("write-mode images not supported in "
                        "automatic testing")

            fill_rand(storage_array)

            if isinstance(arg, ImageArg):
                # must be contiguous
                pre_run_ary = pre_run_storage_array = storage_array.copy()

                ref_args[arg.name] = cl.image_from_array(
                        queue.context, ary.get())
            else:
                pre_run_storage_array = storage_array.copy()

                ary = cl_array.as_strided(storage_array, shape, numpy_strides)
                pre_run_ary = cl_array.as_strided(
                        pre_run_storage_array, shape, numpy_strides)
                ref_args[arg.name] = ary

            ref_arg_data.append(
                    TestArgInfo(
                        name=arg.name,
                        ref_array=ary,
                        ref_storage_array=storage_array,

                        ref_pre_run_array=pre_run_ary,
                        ref_pre_run_storage_array=pre_run_storage_array,

                        ref_shape=shape,
                        ref_strides=strides,
                        ref_alloc_size=alloc_size,
                        ref_numpy_strides=numpy_strides,
                        needs_checking=is_output))

        elif arg.arg_class is TemporaryVariable:
            # global temporary, handled by invocation logic
            pass

        else:
            raise LoopyError("arg type %s not understood" % type(arg))

    return ref_args, ref_arg_data

# }}}


# {{{ "full-scale" arguments

def make_args(kernel, queue, ref_arg_data, parameters):
    import pyopencl as cl
    import pyopencl.array as cl_array

    from loopy.kernel.data import ValueArg, ArrayArg, ImageArg, ConstantArg

    from pymbolic import evaluate

    args = {}
    for arg, arg_desc in zip(kernel.args, ref_arg_data):
        if isinstance(arg, ValueArg):
            arg_value = parameters[arg.name]

            try:
                argv_dtype = arg_value.dtype
            except AttributeError:
                argv_dtype = None

            if argv_dtype != arg.dtype:
                arg_value = arg.dtype.numpy_dtype.type(arg_value)

            args[arg.name] = arg_value

        elif isinstance(arg, ImageArg):
            if arg.name in kernel.get_written_variables():
                raise NotImplementedError("write-mode images not supported in "
                        "automatic testing")

            shape = evaluate_shape(arg.shape, parameters)
            assert shape == arg_desc.ref_shape

            # must be contiguous
            args[arg.name] = cl.image_from_array(
                    queue.context, arg_desc.ref_pre_run_array.get())

        elif isinstance(arg, (ArrayArg, ConstantArg)):
            shape = evaluate(arg.shape, parameters)
            strides = evaluate(get_strides(arg), parameters)

            dtype = arg.dtype
            itemsize = dtype.itemsize
            numpy_strides = [itemsize*s for s in strides]

            alloc_size = sum(astrd*(alen-1) if astrd != 0 else alen-1
                    for alen, astrd in zip(shape, strides)) + 1

            # use contiguous array to transfer to host
            host_ref_contig_array = arg_desc.ref_pre_run_storage_array.get()

            # use device shape/strides
            from pyopencl.compyte.array import as_strided
            host_ref_array = as_strided(host_ref_contig_array,
                    arg_desc.ref_shape, arg_desc.ref_numpy_strides)

            # flatten the thing
            host_ref_flat_array = host_ref_array.flatten()

            # create host array with test shape (but not strides)
            host_contig_array = np.empty(shape, dtype=dtype)

            common_len = min(
                    len(host_ref_flat_array),
                    len(host_contig_array.ravel()))
            host_contig_array.ravel()[:common_len] = \
                    host_ref_flat_array[:common_len]

            # create host array with test shape and storage layout
            host_storage_array = np.empty(alloc_size, dtype)
            host_array = as_strided(
                    host_storage_array, shape, numpy_strides)
            host_array[...] = host_contig_array

            host_contig_array = arg_desc.ref_storage_array.get()
            storage_array = cl_array.to_device(queue, host_storage_array)
            ary = cl_array.as_strided(storage_array, shape, numpy_strides)

            args[arg.name] = ary

            arg_desc.test_storage_array = storage_array
            arg_desc.test_array = ary
            arg_desc.test_shape = shape
            arg_desc.test_strides = strides
            arg_desc.test_numpy_strides = numpy_strides
            arg_desc.test_alloc_size = alloc_size

        else:
            raise LoopyError("arg type not understood")

    return args

# }}}

# }}}


# {{{ default array comparison

def _default_check_result(result, ref_result):
    if not is_dtype_supported(result.dtype) and not (result == ref_result).all():
        return (False, "results do not match exactly")

    if not np.allclose(ref_result, result, rtol=1e-3, atol=1e-3):
        l2_err = (
                np.sum(np.abs(ref_result-result)**2)
                / np.sum(np.abs(ref_result)**2))
        linf_err = (
                np.max(np.abs(ref_result-result))
                / np.max(np.abs(ref_result-result)))
        # pylint: disable=bad-string-format-type
        return (False,
                # pylint: disable=bad-string-format-type
                "results do not match -- (rel) l_2 err: %g, l_inf err: %g"
                % (l2_err, linf_err))
    else:
        return True, None

# }}}


# {{{ find device for reference test

def _enumerate_cl_devices_for_ref_test(blacklist_ref_vendors, need_image_support):
    import pyopencl as cl

    noncpu_devs = []
    cpu_devs = []

    if isinstance(blacklist_ref_vendors, str):
        blacklist_ref_vendors = blacklist_ref_vendors.split(",")

    for pf in cl.get_platforms():
        for dev in pf.get_devices():
            if any(bl in dev.platform.vendor
                    for bl in blacklist_ref_vendors):
                continue

            if need_image_support:
                if not dev.image_support:
                    continue
                if pf.vendor == "The pocl project":
                    # Hahaha, no.
                    continue

            if dev.type & cl.device_type.CPU:
                cpu_devs.append(dev)
            else:
                noncpu_devs.append(dev)

    if not (cpu_devs or noncpu_devs):
        raise LoopyError("no CL device found for test")

    if not cpu_devs:
        warn("No CPU device found for running reference kernel. The reference "
                "computation will either fail because of a timeout "
                "or take a *very* long time.")

    for dev in cpu_devs:
        yield dev

    for dev in noncpu_devs:
        yield dev

# }}}


# {{{ main automatic testing entrypoint

def auto_test_vs_ref(
        ref_prog, ctx, test_prog=None, op_count=(), op_label=(), parameters=None,
        print_ref_code=False, print_code=True, warmup_rounds=2,
        dump_binary=False,
        fills_entire_output=None, do_check=True, check_result=None,
        max_test_kernel_count=1,
        quiet=False, blacklist_ref_vendors=(), ref_entrypoint=None,
        test_entrypoint=None):
    """Compare results of `ref_knl` to the kernels generated by
    scheduling *test_knl*.

    :arg check_result: a callable with :class:`numpy.ndarray` arguments
        *(result, reference_result)* returning a a tuple (class:`bool`,
        message) indicating correctness/acceptability of the result
    :arg max_test_kernel_count: Stop testing after this many *test_knl*
    """
    if parameters is None:
        parameters = {}

    import pyopencl as cl

    if test_prog is None:
        test_prog = ref_prog
        do_check = False

    if ref_entrypoint is None:
        if len(ref_prog.entrypoints) != 1:
            raise LoopyError("Unable to guess entrypoint for ref_prog.")
        ref_entrypoint = list(ref_prog.entrypoints)[0]

    if test_entrypoint is None:
        if len(test_prog.entrypoints) != 1:
            raise LoopyError("Unable to guess entrypoint for ref_prog.")
        test_entrypoint = list(test_prog.entrypoints)[0]

    ref_prog = lp.preprocess_kernel(ref_prog)
    test_prog = lp.preprocess_kernel(test_prog)

    if len(ref_prog[ref_entrypoint].args) != len(test_prog[test_entrypoint].args):
        raise LoopyError("ref_prog and test_prog do not have the same number "
                "of arguments")

    for i, (ref_arg, test_arg) in enumerate(zip(ref_prog[ref_entrypoint].args,
            test_prog[test_entrypoint].args)):
        if ref_arg.name != test_arg.name:
            raise LoopyError("ref_prog and test_prog argument lists disagree at "
                    "index %d (1-based)" % (i+1))

        if ref_arg.dtype != test_arg.dtype:
            raise LoopyError("ref_prog and test_prog argument lists disagree at "
                    "index %d (1-based)" % (i+1))

    from loopy.target.execution import get_highlighted_code

    if isinstance(op_count, (int, float)):
        warn("op_count should be a list", stacklevel=2)
        op_count = [op_count]
    if isinstance(op_label, str):
        warn("op_label should be a list", stacklevel=2)
        op_label = [op_label]

    from time import time

    if check_result is None:
        check_result = _default_check_result

    if fills_entire_output is not None:
        warn("fills_entire_output is deprecated", DeprecationWarning, stacklevel=2)

    # {{{ compile and run reference code

    from loopy.type_inference import infer_unknown_types
    ref_prog = infer_unknown_types(ref_prog, expect_completion=True)

    found_ref_device = False

    ref_errors = []

    from loopy.kernel.data import ImageArg
    need_ref_image_support = any(isinstance(arg, ImageArg)
                                 for arg in ref_prog[ref_entrypoint].args)

    for dev in _enumerate_cl_devices_for_ref_test(
            blacklist_ref_vendors, need_ref_image_support):

        ref_ctx = cl.Context([dev])
        ref_queue = cl.CommandQueue(ref_ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE)
        ref_codegen_result = lp.generate_code_v2(ref_prog)

        logger.info("{} (ref): trying {} for the reference calculation".format(
            ref_entrypoint, dev))

        if not quiet and print_ref_code:
            print(75*"-")
            print("Reference Code:")
            print(75*"-")
            print(get_highlighted_code(
                ref_codegen_result.device_code()))
            print(75*"-")

        try:
            ref_args, ref_arg_data = \
                    make_ref_args(ref_prog[ref_entrypoint], ref_queue, parameters)
            ref_args["out_host"] = False
        except cl.RuntimeError as e:
            if e.code == cl.status_code.IMAGE_FORMAT_NOT_SUPPORTED:
                import traceback
                ref_errors.append("\n".join([
                    75*"-",
                    "On %s:" % dev,
                    75*"-",
                    traceback.format_exc(),
                    75*"-"]))

                continue
            else:
                raise

        found_ref_device = True

        if not do_check:
            break

        ref_queue.finish()

        logger.info("{} (ref): using {} for the reference calculation".format(
            ref_entrypoint, dev))
        logger.info("%s (ref): run" % ref_entrypoint)

        ref_start = time()

        if not AUTO_TEST_SKIP_RUN:
            ref_evt, _ = ref_prog(ref_queue, **ref_args)
        else:
            ref_evt = cl.enqueue_marker(ref_queue)

        ref_queue.finish()
        ref_stop = time()
        ref_elapsed_wall = ref_stop-ref_start

        logger.info("%s (ref): run done" % ref_entrypoint)

        ref_evt.wait()
        ref_elapsed_event = 1e-9*(ref_evt.profile.END-ref_evt.profile.START)

        break

    if not found_ref_device:
        raise LoopyError("could not find a suitable device for the "
                "reference computation.\n"
                "These errors were encountered:\n"+"\n".join(ref_errors))

    # }}}

    # {{{ compile and run parallel code

    need_check = do_check

    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    from loopy.kernel import KernelState
    from loopy.target.pyopencl import PyOpenCLTarget
    if test_prog[test_entrypoint].state not in [
            KernelState.PREPROCESSED,
            KernelState.LINEARIZED]:
        if isinstance(test_prog.target, PyOpenCLTarget):
            test_prog = test_prog.copy(target=PyOpenCLTarget(ctx.devices[0]))

        test_prog = lp.preprocess_kernel(test_prog)

    from loopy.type_inference import infer_unknown_types

    test_prog = infer_unknown_types(test_prog, expect_completion=True)
    test_prog_codegen_result = lp.generate_code_v2(test_prog)

    args = make_args(test_prog[test_entrypoint],
            queue, ref_arg_data, parameters)
    args["out_host"] = False

    if not quiet:
        print(75*"-")
        print("Kernel:")
        print(75*"-")
        if print_code:
            print(get_highlighted_code(
                test_prog_codegen_result.device_code()))
            print(75*"-")
        if dump_binary:
            print(type(test_prog_codegen_result.cl_program))
            print(test_prog_codegen_result.cl_program.binaries[0])
            print(75*"-")

    logger.info("%s: run warmup" % (test_entrypoint))

    for _i in range(warmup_rounds):
        if not AUTO_TEST_SKIP_RUN:
            test_prog(queue, **args)

        if need_check and not AUTO_TEST_SKIP_RUN:
            for arg_desc in ref_arg_data:
                if arg_desc is None:
                    continue
                if not arg_desc.needs_checking:
                    continue

                from pyopencl.compyte.array import as_strided
                ref_ary = as_strided(
                        arg_desc.ref_storage_array.get(),
                        shape=arg_desc.ref_shape,
                        strides=arg_desc.ref_numpy_strides).flatten()
                test_ary = as_strided(
                        arg_desc.test_storage_array.get(),
                        shape=arg_desc.test_shape,
                        strides=arg_desc.test_numpy_strides).flatten()
                common_len = min(len(ref_ary), len(test_ary))
                ref_ary = ref_ary[:common_len]
                test_ary = test_ary[:common_len]

                error_is_small, error = check_result(test_ary, ref_ary)
                if not error_is_small:
                    raise AutomaticTestFailure(error)

                need_check = False

    events = []
    queue.finish()

    logger.info("%s: warmup done" % (test_entrypoint))

    logger.info("%s: timing run" % (test_entrypoint))

    timing_rounds = max(warmup_rounds, 1)

    while True:
        from time import time
        start_time = time()

        evt_start = cl.enqueue_marker(queue)

        for _i in range(timing_rounds):
            if not AUTO_TEST_SKIP_RUN:
                evt, _ = test_prog(queue, **args)
                events.append(evt)
            else:
                events.append(cl.enqueue_marker(queue))

        evt_end = cl.enqueue_marker(queue)

        queue.finish()
        stop_time = time()

        for evt in events:
            evt.wait()
        evt_start.wait()
        evt_end.wait()

        elapsed_event = (1e-9*events[-1].profile.END
                - 1e-9*events[0].profile.START) \
                / timing_rounds
        try:
            elapsed_event_marker = ((1e-9*evt_end.profile.START
                        - 1e-9*evt_start.profile.START)
                    / timing_rounds)
        except cl.RuntimeError:
            elapsed_event_marker = None

        elapsed_wall = (stop_time-start_time)/timing_rounds

        if elapsed_wall * timing_rounds < 0.3:
            timing_rounds *= 4
        else:
            break

    logger.info("%s: timing run done" % (test_entrypoint))

    rates = ""
    for cnt, lbl in zip(op_count, op_label):
        rates += " {:g} {}/s".format(cnt/elapsed_wall, lbl)

    if not quiet:
        def format_float_or_none(v):
            if v is None:
                return "<unavailable>"
            else:
                return "%g" % v

        print("elapsed: %s s event, %s s marker-event %s s wall "
                "(%d rounds)%s" % (
                    format_float_or_none(elapsed_event),
                    format_float_or_none(elapsed_event_marker),
                    format_float_or_none(elapsed_wall), timing_rounds, rates))

    if do_check:
        ref_rates = ""
        for cnt, lbl in zip(op_count, op_label):
            rates += " {:g} {}/s".format(cnt/elapsed_wall, lbl)

        if not quiet:
            print("elapsed: %s s event, %s s marker-event %s s wall "
                    "(%d rounds)%s" % (
                        format_float_or_none(elapsed_event),
                        format_float_or_none(elapsed_event_marker),
                        format_float_or_none(elapsed_wall), timing_rounds, rates))

        if do_check:
            ref_rates = ""
            for cnt, lbl in zip(op_count, op_label):
                ref_rates += " {:g} {}/s".format(cnt/ref_elapsed_event, lbl)
            if not quiet:
                print("ref: elapsed: {:g} s event, {:g} s wall{}".format(
                        ref_elapsed_event, ref_elapsed_wall, ref_rates))

    # }}}

    result_dict = {}
    result_dict["elapsed_event"] = elapsed_event
    result_dict["elapsed_event_marker"] = elapsed_event_marker
    result_dict["elapsed_wall"] = elapsed_wall
    result_dict["timing_rounds"] = timing_rounds

    if do_check:
        result_dict["ref_elapsed_event"] = ref_elapsed_event
        result_dict["ref_elapsed_wall"] = ref_elapsed_wall

    return result_dict


# }}}

# vim: foldmethod=marker

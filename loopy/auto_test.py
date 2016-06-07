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

from six.moves import range, zip
from pytools import Record
from warnings import warn

import numpy as np

import loopy as lp
from loopy.diagnostic import LoopyError, AutomaticTestFailure


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


class TestArgInfo(Record):
    pass


# {{{ "reference" arguments

def make_ref_args(kernel, impl_arg_info, queue, parameters):
    import pyopencl as cl
    import pyopencl.array as cl_array

    from loopy.kernel.data import ValueArg, GlobalArg, ImageArg, TemporaryVariable

    from pymbolic import evaluate

    ref_args = {}
    ref_arg_data = []

    for arg in impl_arg_info:
        kernel_arg = kernel.impl_arg_to_arg.get(arg.name)

        if arg.arg_class is ValueArg:
            if arg.offset_for_name:
                continue

            arg_value = parameters[arg.name]

            try:
                argv_dtype = arg_value.dtype
            except AttributeError:
                argv_dtype = None

            if argv_dtype != arg.dtype:
                arg_value = arg.dtype.numpy_dtype.type(arg_value)

            ref_args[arg.name] = arg_value

            ref_arg_data.append(None)

        elif arg.arg_class is GlobalArg or arg.arg_class is ImageArg:
            if arg.shape is None or any(saxis is None for saxis in arg.shape):
                raise LoopyError("array '%s' needs known shape to use automatic "
                        "testing" % arg.name)

            shape = evaluate_shape(arg.unvec_shape, parameters)
            dtype = kernel_arg.dtype

            is_output = arg.base_name in kernel.get_written_variables()

            if arg.arg_class is ImageArg:
                storage_array = ary = cl_array.empty(
                        queue, shape, dtype, order="C")
                numpy_strides = None
                alloc_size = None
                strides = None
            else:
                strides = evaluate(arg.unvec_strides, parameters)

                from pytools import all
                assert all(s > 0 for s in strides)
                alloc_size = sum(astrd*(alen-1)
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

            if is_output and arg.arg_class is ImageArg:
                raise LoopyError("write-mode images not supported in "
                        "automatic testing")

            fill_rand(storage_array)

            if arg.arg_class is ImageArg:
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
            raise LoopyError("arg type not understood")

    return ref_args, ref_arg_data

# }}}


# {{{ "full-scale" arguments

def make_args(kernel, impl_arg_info, queue, ref_arg_data, parameters):
    import pyopencl as cl
    import pyopencl.array as cl_array

    from loopy.kernel.data import ValueArg, GlobalArg, ImageArg, TemporaryVariable

    from pymbolic import evaluate

    args = {}
    for arg, arg_desc in zip(impl_arg_info, ref_arg_data):
        kernel_arg = kernel.impl_arg_to_arg.get(arg.name)

        if arg.arg_class is ValueArg:
            arg_value = parameters[arg.name]

            try:
                argv_dtype = arg_value.dtype
            except AttributeError:
                argv_dtype = None

            if argv_dtype != arg.dtype:
                arg_value = arg.dtype.numpy_dtype.type(arg_value)

            args[arg.name] = arg_value

        elif arg.arg_class is ImageArg:
            if arg.name in kernel.get_written_variables():
                raise NotImplementedError("write-mode images not supported in "
                        "automatic testing")

            shape = evaluate_shape(arg.unvec_shape, parameters)
            assert shape == arg_desc.ref_shape

            # must be contiguous
            args[arg.name] = cl.image_from_array(
                    queue.context, arg_desc.ref_pre_run_array.get())

        elif arg.arg_class is GlobalArg:
            shape = evaluate(arg.unvec_shape, parameters)
            strides = evaluate(arg.unvec_strides, parameters)

            dtype = kernel_arg.dtype
            itemsize = dtype.itemsize
            numpy_strides = [itemsize*s for s in strides]

            assert all(s > 0 for s in strides)
            alloc_size = sum(astrd*(alen-1)
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

        elif arg.arg_class is TemporaryVariable:
            # global temporary, handled by invocation logic
            pass

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
                /
                np.sum(np.abs(ref_result)**2))
        linf_err = (
                np.max(np.abs(ref_result-result))
                /
                np.max(np.abs(ref_result-result)))
        return (False,
                "results do not match -- (rel) l_2 err: %g, l_inf err: %g"
                % (l2_err, linf_err))
    else:
        return True, None

# }}}


# {{{ find device for reference test

def _enumerate_cl_devices_for_ref_test(blacklist_ref_vendors):
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

            if dev.type & cl.device_type.CPU:
                if "Intel" in dev.platform.vendor:
                    # Sorry, Intel, your CPU CL has gotten too crashy of late.
                    # (Feb 2016)
                    continue

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
        ref_knl, ctx, test_knl=None, op_count=[], op_label=[], parameters={},
        print_ref_code=False, print_code=True, warmup_rounds=2,
        dump_binary=False,
        fills_entire_output=None, do_check=True, check_result=None,
        max_test_kernel_count=1,
        quiet=False, blacklist_ref_vendors=[]):
    """Compare results of `ref_knl` to the kernels generated by
    scheduling *test_knl*.

    :arg check_result: a callable with :class:`numpy.ndarray` arguments
        *(result, reference_result)* returning a a tuple (class:`bool`,
        message) indicating correctness/acceptability of the result
    :arg max_test_kernel_count: Stop testing after this many *test_knl*
    """

    import pyopencl as cl

    if test_knl is None:
        test_knl = ref_knl
        do_check = False

    if len(ref_knl.args) != len(test_knl.args):
        raise LoopyError("ref_knl and test_knl do not have the same number "
                "of arguments")

    for i, (ref_arg, test_arg) in enumerate(zip(ref_knl.args, test_knl.args)):
        if ref_arg.name != test_arg.name:
            raise LoopyError("ref_knl and test_knl argument lists disagree at index "
                    "%d (1-based)" % (i+1))

        if ref_arg.dtype != test_arg.dtype:
            raise LoopyError("ref_knl and test_knl argument lists disagree at index "
                    "%d (1-based)" % (i+1))

    from loopy.compiled import CompiledKernel, get_highlighted_cl_code

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

    from loopy.preprocess import infer_unknown_types
    ref_knl = infer_unknown_types(ref_knl, expect_completion=True)

    found_ref_device = False

    ref_errors = []

    for dev in _enumerate_cl_devices_for_ref_test(blacklist_ref_vendors):
        ref_ctx = cl.Context([dev])
        ref_queue = cl.CommandQueue(ref_ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE)

        pp_ref_knl = lp.preprocess_kernel(ref_knl)

        for knl in lp.generate_loop_schedules(pp_ref_knl):
            ref_sched_kernel = knl
            break

        logger.info("%s (ref): trying %s for the reference calculation" % (
            ref_knl.name, dev))

        ref_compiled = CompiledKernel(ref_ctx, ref_sched_kernel)
        if not quiet and print_ref_code:
            print(75*"-")
            print("Reference Code:")
            print(75*"-")
            print(get_highlighted_cl_code(ref_compiled.code))
            print(75*"-")

        ref_cl_kernel_info = ref_compiled.cl_kernel_info(frozenset())

        try:
            ref_args, ref_arg_data = \
                    make_ref_args(ref_sched_kernel,
                            ref_cl_kernel_info.implemented_data_info,
                            ref_queue, parameters)
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

        logger.info("%s (ref): using %s for the reference calculation" % (
            ref_knl.name, dev))
        logger.info("%s (ref): run" % ref_knl.name)

        ref_start = time()

        if not AUTO_TEST_SKIP_RUN:
            ref_evt, _ = ref_compiled(ref_queue, **ref_args)
        else:
            ref_evt = cl.enqueue_marker(ref_queue)

        ref_queue.finish()
        ref_stop = time()
        ref_elapsed_wall = ref_stop-ref_start

        logger.info("%s (ref): run done" % ref_knl.name)

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

    args = None
    from loopy.kernel import kernel_state
    if test_knl.state not in [
            kernel_state.PREPROCESSED,
            kernel_state.SCHEDULED]:
        test_knl = lp.preprocess_kernel(test_knl)

    if not test_knl.schedule:
        test_kernels = lp.generate_loop_schedules(test_knl)
    else:
        test_kernels = [test_knl]

    test_kernel_count = 0

    from loopy.preprocess import infer_unknown_types
    for i, kernel in enumerate(test_kernels):
        test_kernel_count += 1
        if test_kernel_count > max_test_kernel_count:
            break

        kernel = infer_unknown_types(kernel, expect_completion=True)

        compiled = CompiledKernel(ctx, kernel)

        if args is None:
            cl_kernel_info = compiled.cl_kernel_info(frozenset())

            args = make_args(kernel,
                    cl_kernel_info.implemented_data_info,
                    queue, ref_arg_data, parameters)
        args["out_host"] = False

        if not quiet:
            print(75*"-")
            print("Kernel #%d:" % i)
            print(75*"-")
            if print_code:
                print(compiled.get_highlighted_code())
                print(75*"-")
            if dump_binary:
                print(type(compiled.cl_program))
                print(compiled.cl_program.binaries[0])
                print(75*"-")

        logger.info("%s: run warmup" % (knl.name))

        for i in range(warmup_rounds):
            if not AUTO_TEST_SKIP_RUN:
                compiled(queue, **args)

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

        logger.info("%s: warmup done" % (knl.name))

        logger.info("%s: timing run" % (knl.name))

        timing_rounds = warmup_rounds

        while True:
            from time import time
            start_time = time()

            evt_start = cl.enqueue_marker(queue)

            for i in range(timing_rounds):
                if not AUTO_TEST_SKIP_RUN:
                    evt, _ = compiled(queue, **args)
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

        logger.info("%s: timing run done" % (knl.name))

        rates = ""
        for cnt, lbl in zip(op_count, op_label):
            rates += " %g %s/s" % (cnt/elapsed_wall, lbl)

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
                ref_rates += " %g %s/s" % (cnt/ref_elapsed_event, lbl)
            if not quiet:
                print("ref: elapsed: %g s event, %g s wall%s" % (
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

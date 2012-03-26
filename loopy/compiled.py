from __future__ import division
import pyopencl as cl
import pyopencl.array as cl_array

import numpy as np




# {{{ compiled kernel object

class CompiledKernel:
    def __init__(self, context, kernel, size_args=None, options=[],
             edit_code=False, with_annotation=False):
        self.kernel = kernel
        from loopy.codegen import generate_code
        self.code = generate_code(kernel, with_annotation=with_annotation)

        if edit_code:
            from pytools import invoke_editor
            self.code = invoke_editor(self.code)

        import pyopencl as cl
        try:
            self.cl_program = cl.Program(context, self.code)
            self.cl_kernel = getattr(
                    self.cl_program.build(options=options),
                    kernel.name)
        except:
            print "[Loopy] ----------------------------------------------------"
            print "[Loopy] build failed, here's the source code:"
            print "[Loopy] ----------------------------------------------------"
            print self.code
            print "[Loopy] ----------------------------------------------------"
            print "[Loopy] end source code"
            print "[Loopy] ----------------------------------------------------"
            raise

        from loopy.kernel import ScalarArg

        arg_types = []
        for arg in kernel.args:
            if isinstance(arg, ScalarArg):
                arg_types.append(arg.dtype)
            else:
                arg_types.append(None)

        self.cl_kernel.set_scalar_arg_dtypes(arg_types)

        from pymbolic import compile
        if size_args is None:
            self.size_args = kernel.scalar_loop_args
        else:
            self.size_args = size_args

        gsize_expr, lsize_expr = kernel.get_grid_sizes_as_exprs()

        if not gsize_expr: gsize_expr = (1,)
        if not lsize_expr: lsize_expr = (1,)

        self.global_size_func = compile(
                gsize_expr, self.size_args)
        self.local_size_func = compile(
                lsize_expr, self.size_args)

# }}}



def print_highlighted_code(text):
    try:
        from pygments import highlight
    except ImportError:
        print text
    else:
        from pygments.lexers import CLexer
        from pygments.formatters import TerminalFormatter

        print highlight(text, CLexer(), TerminalFormatter())




# {{{ timing driver

def drive_timing_run(kernel_generator, queue, launch, flop_count=None,
        options=[], print_code=True, edit_code=False):

    def time_run(compiled_knl, warmup_rounds=2, timing_rounds=5):
        check = True
        for i in range(warmup_rounds):
            launch(compiled_knl.cl_kernel,
                    compiled.global_size_func, compiled.local_size_func,
                    check=check)
            check = False

        events = []
        for i in range(timing_rounds):
            events.append(
                    launch(compiled_knl.cl_kernel,
                        compiled.global_size_func, compiled.local_size_func,
                        check=check))
        for evt in events:
            evt.wait()

        return sum(1e-9*evt.profile.END-1e-9*evt.profile.START for evt in events)/timing_rounds

    soln_count = 0
    for kernel in kernel_generator:

        compiled = CompiledKernel(queue.context, kernel, options=options,
                edit_code=edit_code)

        print "-----------------------------------------------"
        print "SOLUTION #%d" % soln_count
        print "-----------------------------------------------"
        if print_code:
            print_highlighted_code(compiled.code)
            print "-----------------------------------------------"

        elapsed = time_run(compiled)

        print "time: %f" % elapsed
        if flop_count is not None:
            print "gflops/s: %f (#%d)" % (
                    flop_count/elapsed/1e9, soln_count)
        print "-----------------------------------------------"

        soln_count += 1

    print "%d solutions" % soln_count

# }}}

# {{{ automatic testing

def fill_rand(ary):
    from pyopencl.clrandom import fill_rand
    if ary.dtype.kind == "c":
        real_dtype = ary.dtype.type(0).real.dtype
        real_ary = ary.view(real_dtype)

        fill_rand(real_ary, luxury=0)
    else:
        fill_rand(ary, luxury=0)




def make_ref_args(kernel, queue, parameters,
        fill_value):
    from loopy.kernel import ScalarArg, ArrayArg, ImageArg

    from pymbolic import evaluate

    result = []
    input_arrays = []
    output_arrays = []

    for arg in kernel.args:
        if isinstance(arg, ScalarArg):
            arg_value = parameters[arg.name]

            try:
                argv_dtype = arg_value.dtype
            except AttributeError:
                argv_dtype = None

            if argv_dtype != arg.dtype:
                arg_value = arg.dtype.type(arg_value)

            result.append(arg_value)

        elif isinstance(arg, (ArrayArg, ImageArg)):
            if arg.shape is None:
                raise ValueError("arrays need known shape to use automatic "
                        "testing")

            shape = evaluate(arg.shape, parameters)
            if isinstance(arg, ImageArg):
                order = "C"
            else:
                order = arg.order
                assert arg.offset == 0

            ary = cl_array.empty(queue, shape, arg.dtype, order=order)
            if arg.name in kernel.get_written_variables():
                if isinstance(arg, ImageArg):
                    raise RuntimeError("write-mode images not supported in "
                            "automatic testing")

                if arg.dtype.isbuiltin:
                    ary.fill(fill_value)
                else:
                    from warnings import warn
                    warn("Cannot pre-fill array of dtype '%s'" % arg.dtype)

                output_arrays.append(ary)
                result.append(ary.data)
            else:
                fill_rand(ary)
                input_arrays.append(ary)
                if isinstance(arg, ImageArg):
                    result.append(cl.image_from_array(queue.context, ary.get(), 1))
                else:
                    result.append(ary.data)

        else:
            raise RuntimeError("arg type not understood")

    return result, input_arrays, output_arrays




def make_args(queue, kernel, ref_input_arrays, parameters,
        fill_value):
    from loopy.kernel import ScalarArg, ArrayArg, ImageArg

    from pymbolic import evaluate

    result = []
    output_arrays = []
    for arg in kernel.args:
        if isinstance(arg, ScalarArg):
            arg_value = parameters[arg.name]

            try:
                argv_dtype = arg_value.dtype
            except AttributeError:
                argv_dtype = None

            if argv_dtype != arg.dtype:
                arg_value = arg.dtype.type(arg_value)

            result.append(arg_value)

        elif isinstance(arg, (ArrayArg, ImageArg)):
            if arg.name in kernel.get_written_variables():
                if isinstance(arg, ImageArg):
                    raise RuntimeError("write-mode images not supported in "
                            "automatic testing")

                shape = evaluate(arg.shape, parameters)
                ary = cl_array.empty(queue, shape, arg.dtype, order=arg.order)

                if arg.dtype.isbuiltin:
                    ary.fill(fill_value)
                else:
                    from warnings import warn
                    warn("Cannot pre-fill array of dtype '%s'" % arg.dtype)

                assert arg.offset == 0
                output_arrays.append(ary)
                result.append(ary.data)
            else:
                ref_arg = ref_input_arrays.pop(0)

                if isinstance(arg, ImageArg):
                    result.append(cl.image_from_array(queue.context, ref_arg.get(), 1))
                else:
                    ary = cl_array.to_device(queue, ref_arg.get())
                    result.append(ary.data)

        else:
            raise RuntimeError("arg type not understood")

    return result, output_arrays




def _default_check_result(result, ref_result):
    if not np.allclose(ref_result, result, rtol=1e-3, atol=1e-3):
        l2_err = np.sum(np.abs(ref_result-result)**2)/np.sum(np.abs(ref_result)**2)
        linf_err = np.max(np.abs(ref_result-result))/np.max(np.abs(ref_result-result))
        return (False,
                "results do not match(rel) l_2 err: %g, l_inf err: %g"
                % (l2_err, linf_err))
    else:
        return True, None




def auto_test_vs_ref(ref_knl, ctx, kernel_gen, op_count, op_label, parameters,
        print_ref_code=False, print_code=True, warmup_rounds=2,
        edit_code=False, dump_binary=False, with_annotation=False,
        fills_entire_output=True, check_result=None):
    """
    :arg check_result: a callable with :cls:`numpy.ndarray` arguments
        *(result, reference_result)* returning a a tuple (class:`bool`, message)
        indicating correctness/acceptability of the result
    """
    from time import time

    if check_result is None:
        check_result = _default_check_result

    if fills_entire_output:
        fill_value_ref = -17
        fill_value = -18
    else:
        fill_value_ref = -17
        fill_value = fill_value_ref

    # {{{ set up CL context for reference run
    last_dev = None
    last_cpu_dev = None

    for pf in cl.get_platforms():
        for dev in pf.get_devices():
            last_dev  = dev
            if dev.type == cl.device_type.CPU:
                last_cpu_dev = dev

    if last_cpu_dev is None:
        dev = last_dev
        from warnings import warn
        warn("No CPU device found for reference test, using %s." % dev)
    else:
        dev = last_cpu_dev

    print "using %s for the reference calculation" % dev

    # }}}

    # {{{ compile and run reference code

    ref_ctx = cl.Context([dev])
    ref_queue = cl.CommandQueue(ref_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    import loopy as lp
    ref_kernel_gen = lp.generate_loop_schedules(ref_knl)
    for knl in lp.check_kernels(ref_kernel_gen, parameters):
        ref_sched_kernel = knl
        break

    ref_compiled = CompiledKernel(ref_ctx, ref_sched_kernel,
            with_annotation=with_annotation)
    if print_ref_code:
        print 75*"-"
        print "Reference Code:"
        print 75*"-"
        print_highlighted_code(ref_compiled.code)
        print 75*"-"

    ref_args, ref_input_arrays, ref_output_arrays = \
            make_ref_args(ref_sched_kernel, ref_queue, parameters,
                    fill_value=fill_value_ref)

    ref_queue.finish()
    ref_start = time()

    domain_parameters = dict((name, parameters[name])
            for name in ref_knl.scalar_loop_args)

    ref_evt = ref_compiled.cl_kernel(ref_queue,
            ref_compiled.global_size_func(**domain_parameters),
            ref_compiled.local_size_func(**domain_parameters),
            *ref_args,
            g_times_l=True)

    ref_queue.finish()
    ref_stop = time()
    ref_elapsed_wall = ref_stop-ref_start

    ref_evt.wait()
    ref_elapsed = 1e-9*(ref_evt.profile.END-ref_evt.profile.SUBMIT)

    # }}}

    # {{{ compile and run parallel code

    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    args = None
    for i, kernel in enumerate(kernel_gen):
        if args is None:
            args, output_arrays = make_args(queue, kernel, ref_input_arrays, parameters,
                    fill_value=fill_value)

        compiled = CompiledKernel(ctx, kernel, edit_code=edit_code,
                with_annotation=with_annotation)

        print 75*"-"
        print "Kernel #%d:" % i
        print 75*"-"
        if print_code:
            print_highlighted_code(compiled.code)
            print 75*"-"
        if dump_binary:
            print type(compiled.cl_program)
            print compiled.cl_program.binaries[0]
            print 75*"-"

        do_check = True

        gsize = compiled.global_size_func(**domain_parameters)
        lsize = compiled.local_size_func(**domain_parameters)
        for i in range(warmup_rounds):
            evt = compiled.cl_kernel(queue, gsize, lsize, *args, g_times_l=True)

            if do_check:
                for ref_out_ary, out_ary in zip(ref_output_arrays, output_arrays):
                    error_is_small, error = check_result(out_ary.get(), ref_out_ary.get())
                    assert error_is_small, error
                    do_check = False

        events = []
        queue.finish()

        timing_rounds = warmup_rounds

        while True:
            from time import time
            start_time = time()

            evt_start = cl.enqueue_marker(queue)

            for i in range(timing_rounds):
                events.append(
                        compiled.cl_kernel(queue, gsize, lsize, *args, g_times_l=True))

            evt_end = cl.enqueue_marker(queue)

            queue.finish()
            stop_time = time()

            for evt in events:
                evt.wait()
            evt_start.wait()
            evt_end.wait()

            elapsed = (1e-9*events[-1].profile.END-1e-9*events[0].profile.SUBMIT) \
                    / timing_rounds
            try:
                elapsed_evt_2 = "%g" % \
                        ((1e-9*evt_end.profile.START-1e-9*evt_start.profile.START) \
                        / timing_rounds)
            except cl.RuntimeError:
                elapsed_evt_2 = "<unavailable>"

            elapsed_wall = (stop_time-start_time)/timing_rounds

            if elapsed_wall * timing_rounds < 0.3:
                timing_rounds *= 4
            else:
                break

        print "elapsed: %g s event, %s s other-event %g s wall, rate: %g %s/s (%d rounds)" % (
                elapsed, elapsed_evt_2, elapsed_wall, op_count/elapsed, op_label,
                timing_rounds)
        print "ref: elapsed: %g s event, %g s wall, rate: %g %s/s" % (
                ref_elapsed, ref_elapsed_wall, op_count/ref_elapsed, op_label)

    # }}}

from pytools import MovedFunctionDeprecationWrapper

auto_test_vs_seq = MovedFunctionDeprecationWrapper(auto_test_vs_ref)

# }}}

# vim: foldmethod=marker

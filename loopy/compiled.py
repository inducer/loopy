from __future__ import division
import pyopencl as cl
import pyopencl.array as cl_array

import numpy as np




# {{{ argument checking

def _arg_matches_spec(arg, val, other_args):
    import loopy as lp
    if isinstance(arg, lp.GlobalArg):
        from pymbolic import evaluate
        shape = evaluate(arg.shape, other_args)

        if arg.dtype != val.dtype:
            raise TypeError("dtype mismatch on argument '%s' "
                    "(got: %s, expected: %s)"
                    % (arg.name, val.dtype, arg.dtype))
        if shape != val.shape:
            raise TypeError("shape mismatch on argument '%s' "
                    "(got: %s, expected: %s)"
                    % (arg.name, val.shape, shape))
        if arg.order == "F" and not val.flags.f_contiguous:
            raise TypeError("order mismatch on argument '%s' "
                    "(expected Fortran-contiguous, but isn't)"
                    % (arg.name))
        if arg.order == "C" and not val.flags.c_contiguous:
            print id(val), val.flags
            raise TypeError("order mismatch on argument '%s' "
                    "(expected C-contiguous, but isn't)"
                    % (arg.name))

    return True

# }}}

# {{{ compiled kernel object

class CompiledKernel:
    def __init__(self, context, kernel, size_args=None, options=[],
             edit_code=False, codegen_kwargs={}):
        """
        :arg kernel: may be a loopy.LoopKernel, a generator returning kernels
          (a warning will be issued if more than one is returned). If the kernel
          has not yet been loop-scheduled, that is done, too, with no specific
          arguments.
        """
        import loopy as lp

        # {{{ do scheduling, if not yet done

        needs_check = False

        if not isinstance(kernel, lp.LoopKernel) or kernel.schedule is None:
            if isinstance(kernel, lp.LoopKernel):
                # kernel-iterable, really
                kernel = lp.generate_loop_schedules(kernel)

            kernel_count = 0

            for scheduled_kernel in kernel:
                kernel_count += 1

                if kernel_count == 1:
                    # use the first schedule
                    kernel = scheduled_kernel

                if kernel_count == 2:
                    from warnings import warn
                    warn("kernel scheduling was ambiguous--more than one "
                            "schedule found, ignoring", stacklevel=2)
                    break

            needs_check = True

        # Whether we need to call check_kernels. Since we don't have parameter
        # values now, we'll do that on first invocation.

        self.needs_check = needs_check

        # }}}

        self.kernel = kernel
        from loopy.codegen import generate_code
        self.code = generate_code(kernel, **codegen_kwargs)

        if edit_code:
            from pytools import invoke_editor
            self.code = invoke_editor(self.code, "code.cl")

        try:
            self.cl_program = cl.Program(context, self.code)
            self.cl_kernel = getattr(
                    self.cl_program.build(options=options),
                    kernel.name)
        except KeyboardInterrupt:
            raise
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

        if size_args is None:
            self.size_args = kernel.scalar_loop_args
        else:
            self.size_args = size_args

        gsize_expr, lsize_expr = kernel.get_grid_sizes_as_exprs()

        if not gsize_expr: gsize_expr = (1,)
        if not lsize_expr: lsize_expr = (1,)

        from pymbolic import compile
        self.global_size_func = compile(
                gsize_expr, self.size_args)
        self.local_size_func = compile(
                lsize_expr, self.size_args)

    def __call__(self, queue, **kwargs):
        """If all array arguments are :mod:`numpy` arrays, defaults to returning
        numpy arrays as well.
        """

        allocator = kwargs.pop("allocator", None)
        wait_for = kwargs.pop("wait_for", None)
        out_host = kwargs.pop("out_host", None)
        no_run = kwargs.pop("no_run", None)

        import loopy as lp

        if self.needs_check:
            assert len(list(lp.check_kernels([self.kernel], kwargs))) == 1

            self.needs_check = False

        domain_parameters = dict((name, kwargs[name])
                for name in self.kernel.scalar_loop_args)

        args = []
        outputs = []
        encountered_non_numpy = False

        kwargs_copy = kwargs.copy()

        for arg in self.kernel.args:
            is_written = arg.name in self.kernel.get_written_variables()

            val = kwargs_copy.pop(arg.name, None)

            # automatically transfer host-side arrays
            if isinstance(arg, lp.GlobalArg):
                if isinstance(val, np.ndarray):
                    # synchronous, so nothing to worry about
                    val = cl_array.to_device(queue, val, allocator=allocator)
                elif val is not None:
                    encountered_non_numpy = True

            if val is None:
                if not is_written:
                    raise TypeError("must supply input argument '%s'" % arg.name)

                if isinstance(arg, lp.ImageArg):
                    raise RuntimeError("write-mode image '%s' must "
                            "be explicitly supplied" % arg.name)

                from pymbolic import evaluate
                shape = evaluate(arg.shape, kwargs)
                val = cl_array.empty(queue, shape, arg.dtype, order=arg.order,
                        allocator=allocator)
            else:
                assert _arg_matches_spec(arg, val, kwargs)

            if is_written:
                outputs.append(val)

            if isinstance(arg, lp.GlobalArg):
                args.append(val.data)
            else:
                args.append(val)

        assert not kwargs_copy, (
                "extra arguments: "+", ".join(kwargs_copy.iterkeys()))

        if no_run:
            evt = cl.enqueue_marker(queue)
        else:
            evt = self.cl_kernel(queue,
                    self.global_size_func(**domain_parameters),
                    self.local_size_func(**domain_parameters),
                    *args,
                    g_times_l=True, wait_for=wait_for)

        if out_host is None and not encountered_non_numpy:
            out_host = True
        if out_host:
            outputs = [o.get() for o in outputs]

        return evt, outputs

    def print_code(self):
        print get_highlighted_code(self.code)


# }}}




def get_highlighted_code(text):
    try:
        from pygments import highlight
    except ImportError:
        return text
    else:
        from pygments.lexers import CLexer
        from pygments.formatters import TerminalFormatter

        return highlight(text, CLexer(), TerminalFormatter())




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
            print get_highlighted_code(compiled.code)
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
    from loopy.kernel import ScalarArg, GlobalArg, ImageArg

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

        elif isinstance(arg, (GlobalArg, ImageArg)):
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
    from loopy.kernel import ScalarArg, GlobalArg, ImageArg

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

        elif isinstance(arg, (GlobalArg, ImageArg)):
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




def auto_test_vs_ref(ref_knl, ctx, kernel_gen, op_count=[], op_label=[], parameters={},
        print_ref_code=False, print_code=True, warmup_rounds=2,
        edit_code=False, dump_binary=False, codegen_kwargs={},
        fills_entire_output=True, check_result=None):
    """Compare results of `ref_knl` to the kernels generated by the generator
    `kernel_gen`.

    :arg check_result: a callable with :class:`numpy.ndarray` arguments
        *(result, reference_result)* returning a a tuple (class:`bool`, message)
        indicating correctness/acceptability of the result
    """

    if isinstance(op_count, (int, float)):
        from warnings import warn
        warn("op_count should be a list", stacklevel=2)
        op_count = [op_count]
    if isinstance(op_label, str):
        from warnings import warn
        warn("op_label should be a list", stacklevel=2)
        op_label = [op_label]

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
            codegen_kwargs=codegen_kwargs)
    if print_ref_code:
        print 75*"-"
        print "Reference Code:"
        print 75*"-"
        print get_highlighted_code(ref_compiled.code)
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
                codegen_kwargs=codegen_kwargs)

        print 75*"-"
        print "Kernel #%d:" % i
        print 75*"-"
        if print_code:
            print get_highlighted_code(compiled.code)
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

        rates = ""
        for cnt, lbl in zip(op_count, op_label):
            rates += " %g %s/s" % (cnt/elapsed, lbl)

        print "elapsed: %g s event, %s s other-event %g s wall (%d rounds)%s" % (
                elapsed, elapsed_evt_2, elapsed_wall, timing_rounds, rates)

        ref_rates = ""
        for cnt, lbl in zip(op_count, op_label):
            ref_rates += " %g %s/s" % (cnt/ref_elapsed, lbl)
        print "ref: elapsed: %g s event, %g s wall%s" % (
                ref_elapsed, ref_elapsed_wall, ref_rates)

    # }}}

from pytools import MovedFunctionDeprecationWrapper

auto_test_vs_seq = MovedFunctionDeprecationWrapper(auto_test_vs_ref)

# }}}

# vim: foldmethod=marker

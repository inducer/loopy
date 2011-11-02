from __future__ import division
import pyopencl as cl
import pyopencl.array as cl_array

import numpy as np




# {{{ compiled kernel object

class CompiledKernel:
    def __init__(self, context, kernel, size_args=None, options=[],
             edit_code=False):
        self.kernel = kernel
        from loopy.codegen import generate_code
        self.code = generate_code(kernel)

        if edit_code:
            from pytools import invoke_editor
            self.code = invoke_editor(self.code)

        import pyopencl as cl
        try:
            self.cl_kernel = getattr(
                    cl.Program(context, self.code).build(options=options),
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

def make_seq_args(kernel, queue, parameters):
    from loopy.kernel import ScalarArg, ArrayArg

    from pymbolic import evaluate

    result = []
    input_arrays = []
    output_arrays = []

    for arg in kernel.args:
        if isinstance(arg, ScalarArg):
            result.append(arg.dtype(parameters[arg.name]))

        elif isinstance(arg, ArrayArg):
            if arg.shape is None:
                raise ValueError("arrays need known shape to use automatic "
                        "testing")

            shape = evaluate(arg.shape, parameters)
            ary = cl_array.empty(queue, shape, arg.dtype, order=arg.order)
            assert arg.offset == 0
            if arg.name in kernel.get_written_variables():
                ary.fill(-17)
                output_arrays.append(ary)
            else:
                from pyopencl.clrandom import fill_rand
                fill_rand(ary, luxury=2)
                input_arrays.append(ary)
            result.append(ary.data)

        else:
            raise RuntimeError("arg type not understood")

    return result, input_arrays, output_arrays




def make_args(queue, kernel, seq_input_arrays, parameters):
    from loopy.kernel import ScalarArg, ArrayArg

    from pymbolic import evaluate

    result = []
    output_arrays = []
    for arg in kernel.args:
        if isinstance(arg, ScalarArg):
            result.append(arg.dtype(parameters[arg.name]))

        if isinstance(arg, ArrayArg):
            if arg.name in kernel.get_written_variables():
                shape = evaluate(arg.shape, parameters)
                ary = cl_array.empty(queue, shape, arg.dtype, order=arg.order)
                assert arg.offset == 0
                output_arrays.append(ary)
            else:
                seq_arg = seq_input_arrays.pop(0)
                ary = cl_array.to_device(queue, seq_arg.get())

            result.append(ary.data)

    return result, output_arrays




def auto_test_vs_seq(seq_knl, ctx, kernel_gen, op_count, op_label, parameters,
        print_seq_code=False, print_code=True):
    # {{{ set up CL context for sequential run
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
        warn("No CPU device found for sequential test, using %s." % dev)
    else:
        dev = last_cpu_dev

    print "using", dev

    # }}}

    # {{{ compile and run sequential code

    seq_ctx = cl.Context([dev])
    seq_queue = cl.CommandQueue(seq_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    import loopy as lp
    seq_kernel_gen = lp.generate_loop_schedules(seq_knl)
    for knl in lp.check_kernels(seq_kernel_gen, {}):
        seq_sched_kernel = knl
        break

    seq_compiled = CompiledKernel(seq_ctx, seq_sched_kernel)
    if print_seq_code:
        print "----------------------------------------------------------"
        print "Sequential Code:"
        print "----------------------------------------------------------"
        print_highlighted_code(seq_compiled.code)
        print "----------------------------------------------------------"

    seq_args, seq_input_arrays, seq_output_arrays = \
            make_seq_args(seq_sched_kernel, seq_queue, parameters)

    seq_evt = seq_compiled.cl_kernel(seq_queue,
            seq_compiled.global_size_func(**parameters),
            seq_compiled.local_size_func(**parameters),
            *seq_args,
            g_times_l=True)

    # }}}

    # {{{ compile and run parallel code

    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    args = None
    for i, kernel in enumerate(kernel_gen):
        if args is None:
            args, output_arrays = make_args(queue, kernel, seq_input_arrays, parameters)

        compiled = CompiledKernel(ctx, kernel)
        if print_code:
            print "----------------------------------------------------------"
            print "Kernel #%d:" % i
            print "----------------------------------------------------------"
            print_highlighted_code(compiled.code)
            print "----------------------------------------------------------"

        evt = compiled.cl_kernel(queue,
                compiled.global_size_func(**parameters),
                compiled.local_size_func(**parameters),
                *args,
                g_times_l=True)

        for seq_out_ary, out_ary in zip(seq_output_arrays, output_arrays):
            assert np.allclose(seq_out_ary.get(), out_ary.get())

    # }}}


# }}}

# vim foldmethod=marker

from __future__ import division




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
            self.size_args = [arg.name for arg in kernel.args if isinstance(arg, ScalarArg)]
        else:
            self.size_args = size_args

        from loopy.kernel import TAG_GROUP_IDX, TAG_WORK_ITEM_IDX
        gsize_expr = tuple(self.kernel.tag_type_lengths(
            TAG_GROUP_IDX, allow_parameters=True))
        lsize_expr = tuple(self.kernel.tag_type_lengths(
            TAG_WORK_ITEM_IDX, allow_parameters=False))

        if not gsize_expr: gsize_expr = (1,)
        if not lsize_expr: lsize_expr = (1,)

        self.global_size_func = compile(
                gsize_expr, self.size_args)
        self.local_size_func = compile(
                lsize_expr, self.size_args)

# }}}




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
            print compiled.code
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

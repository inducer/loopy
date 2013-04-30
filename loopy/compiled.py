from __future__ import division

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




import pyopencl as cl
import pyopencl.array as cl_array

import numpy as np

from pytools import Record, memoize_method

AUTO_TEST_SKIP_RUN = False




# {{{ argument checking

def _arg_matches_spec(arg, val, other_args):
    import loopy as lp
    if isinstance(arg, lp.GlobalArg):
        from pymbolic import evaluate
        shape = evaluate(arg.shape, other_args)
        strides = evaluate(arg.numpy_strides, other_args)

        if arg.dtype != val.dtype:
            raise TypeError("dtype mismatch on argument '%s' "
                    "(got: %s, expected: %s)"
                    % (arg.name, val.dtype, arg.dtype))
        if shape != val.shape:
            raise TypeError("shape mismatch on argument '%s' "
                    "(got: %s, expected: %s)"
                    % (arg.name, val.shape, shape))
        if strides != tuple(val.strides):
            raise ValueError("strides mismatch on argument '%s' "
                    "(got: %s, expected: %s)"
                    % (arg.name, val.strides, strides))

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

        self.context = context
        self.kernel = kernel
        self.edit_code = edit_code
        self.codegen_kwargs = codegen_kwargs
        self.options = options

        # {{{ precompile, store grid size functions

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

        # }}}

    @memoize_method
    def get_kernel(self, dtype_mapping_set):
        kernel = self.kernel

        from loopy.kernel.tools import (
                add_argument_dtypes,
                infer_argument_dtypes,
                get_arguments_with_incomplete_dtype)

        if get_arguments_with_incomplete_dtype(kernel):
            if dtype_mapping_set is not None:
                kernel = add_argument_dtypes(kernel, dict(dtype_mapping_set))

            kernel = infer_argument_dtypes(kernel)

            incomplete_args = get_arguments_with_incomplete_dtype(kernel)
            if incomplete_args:
                raise RuntimeError("not all argument dtypes are specified "
                        "or could be inferred: " + ", ".join(incomplete_args))

        return kernel

    @memoize_method
    def get_cl_kernel(self, dtype_mapping_set):
        kernel = self.get_kernel(dtype_mapping_set)

        from loopy.codegen import generate_code
        code = generate_code(kernel, **self.codegen_kwargs)

        if self.edit_code:
            from pytools import invoke_editor
            code = invoke_editor(code, "code.cl")

        try:
            cl_program = cl.Program(self.context, code)
            cl_kernel = getattr(
                    cl_program.build(options=self.options),
                    kernel.name)
        except KeyboardInterrupt:
            raise
        except:
            print "[Loopy] ----------------------------------------------------"
            print "[Loopy] build failed, here's the source code:"
            print "[Loopy] ----------------------------------------------------"
            print code
            print "[Loopy] ----------------------------------------------------"
            print "[Loopy] end source code"
            print "[Loopy] ----------------------------------------------------"
            raise

        from loopy.kernel.data import ValueArg

        arg_types = []
        for arg in kernel.args:
            if isinstance(arg, ValueArg):
                arg_types.append(arg.dtype)
            else:
                arg_types.append(None)

        cl_kernel.set_scalar_arg_dtypes(arg_types)

        return kernel, cl_kernel

    # {{{ debugging aids

    def get_code(self, dtype_dict=None):
        if dtype_dict is not None:
            dtype_dict = frozenset(dtype_dict.items())

        kernel = self.get_kernel(dtype_dict)

        from loopy.codegen import generate_code
        return generate_code(kernel, **self.codegen_kwargs)

    def get_highlighted_code(self, dtype_dict=None):
        return get_highlighted_code(self.get_code(dtype_dict))

    @property
    def code(self):
        from warnings import warn
        warn("CompiledKernel.code is deprecated. Use .get_code() instead.",
                DeprecationWarning, stacklevel=2)

        return self.get_code()

    # }}}

    def __call__(self, queue, **kwargs):
        """If all array arguments are :mod:`numpy` arrays, defaults to returning
        numpy arrays as well.
        """

        allocator = kwargs.pop("allocator", None)
        wait_for = kwargs.pop("wait_for", None)
        out_host = kwargs.pop("out_host", None)
        no_run = kwargs.pop("no_run", None)
        warn_numpy = kwargs.pop("warn_numpy", None)

        # {{{ process arg types, get cl kernel

        dtype_dict = {}
        for arg in self.kernel.args:
            val = kwargs.get(arg.name)
            if val is not None:
                try:
                    dtype = val.dtype
                except AttributeError:
                    pass
                else:
                    dtype_dict[arg.name] = dtype

        kernel, cl_kernel = self.get_cl_kernel(frozenset(dtype_dict.iteritems()))
        del dtype_dict

        # }}}

        import loopy as lp

        domain_parameters = dict((name, int(kwargs[name]))
                for name in kernel.scalar_loop_args)

        args = []
        outputs = []
        encountered_numpy = False

        kwargs_copy = kwargs.copy()

        for arg in kernel.args:
            is_written = arg.name in kernel.get_written_variables()

            val = kwargs_copy.pop(arg.name, None)

            # automatically transfer host-side arrays
            if isinstance(arg, lp.GlobalArg):
                if isinstance(val, np.ndarray):
                    # synchronous, so nothing to worry about
                    val = cl_array.to_device(queue, val, allocator=allocator)
                    encountered_numpy = True
                    if warn_numpy:
                        from warnings import warn
                        warn("argument '%s' was passed as a numpy array, "
                                "performing implicit transfer" % arg.name,
                                stacklevel=2)

            if val is None:
                if not is_written:
                    raise TypeError("must supply input argument '%s'" % arg.name)

                if isinstance(arg, lp.ImageArg):
                    raise RuntimeError("write-mode image '%s' must "
                            "be explicitly supplied" % arg.name)

                from pymbolic import evaluate
                shape = evaluate(arg.shape, kwargs)
                numpy_strides = evaluate(arg.numpy_strides, kwargs)

                from pytools import all
                assert all(s > 0 for s in numpy_strides)
                alloc_size = (sum(astrd*(alen-1)
                        for alen, astrd in zip(shape, numpy_strides))
                        + arg.dtype.itemsize)

                if allocator is None:
                    storage = cl.Buffer(queue.context, cl.mem_flags.READ_WRITE, alloc_size)
                else:
                    storage = allocator(alloc_size)

                val = cl_array.Array(queue, shape, arg.dtype,
                        strides=numpy_strides, data=storage, allocator=allocator)
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
            evt = cl_kernel(queue,
                    self.global_size_func(**domain_parameters),
                    self.local_size_func(**domain_parameters),
                    *args,
                    g_times_l=True, wait_for=wait_for)

        if out_host is None and encountered_numpy:
            out_host = True
        if out_host:
            outputs = [o.get() for o in outputs]

        return evt, outputs

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




# {{{ automatic testing

def fill_rand(ary):
    from pyopencl.clrandom import fill_rand
    if ary.dtype.kind == "c":
        real_dtype = ary.dtype.type(0).real.dtype
        real_ary = ary.view(real_dtype)

        fill_rand(real_ary, luxury=0)
    else:
        fill_rand(ary, luxury=0)





class TestArgInfo(Record):
    pass

def make_ref_args(kernel, queue, parameters,
        fill_value):
    from loopy.kernel.data import ValueArg, GlobalArg, ImageArg

    from pymbolic import evaluate

    ref_args = {}
    arg_descriptors = []

    for arg in kernel.args:
        if isinstance(arg, ValueArg):
            arg_value = parameters[arg.name]

            try:
                argv_dtype = arg_value.dtype
            except AttributeError:
                argv_dtype = None

            if argv_dtype != arg.dtype:
                arg_value = arg.dtype.type(arg_value)

            ref_args[arg.name] = arg_value

            arg_descriptors.append(None)

        elif isinstance(arg, (GlobalArg, ImageArg)):
            if arg.shape is None:
                raise ValueError("arrays need known shape to use automatic "
                        "testing")

            shape = evaluate(arg.shape, parameters)

            is_output = arg.name in kernel.get_written_variables()
            is_image = isinstance(arg, ImageArg)

            if is_image:
                storage_array = ary = cl_array.empty(queue, shape, arg.dtype, order="C")
                numpy_strides = None
                alloc_size = None
                strides = None
            else:
                assert arg.offset == 0

                strides = evaluate(arg.strides, parameters)

                from pytools import all
                assert all(s > 0 for s in strides)
                alloc_size = sum(astrd*(alen-1)
                        for alen, astrd in zip(shape, strides)) + 1

                itemsize = arg.dtype.itemsize
                numpy_strides = [itemsize*s for s in strides]

                storage_array = cl_array.empty(queue, alloc_size, arg.dtype)
                ary = cl_array.as_strided(storage_array, shape, numpy_strides)

            if is_output:
                if is_image:
                    raise RuntimeError("write-mode images not supported in "
                            "automatic testing")

                if arg.dtype.isbuiltin:
                    storage_array.fill(fill_value)
                else:
                    from warnings import warn
                    warn("Cannot pre-fill array of dtype '%s'" % arg.dtype)

                ref_args[arg.name] = ary
            else:
                fill_rand(storage_array)
                if isinstance(arg, ImageArg):
                    # must be contiguous
                    ref_args[arg.name] = cl.image_from_array(queue.context, ary.get())
                else:
                    ref_args[arg.name] = ary

            arg_descriptors.append(
                    TestArgInfo(
                        name=arg.name,
                        ref_array=ary,
                        ref_storage_array=storage_array,
                        ref_shape=shape,
                        ref_strides=strides,
                        ref_alloc_size=alloc_size,
                        ref_numpy_strides=numpy_strides,
                        needs_checking=is_output))
        else:
            raise RuntimeError("arg type not understood")

    return ref_args, arg_descriptors




def make_args(queue, kernel, arg_descriptors, parameters,
        fill_value):
    from loopy.kernel.data import ValueArg, GlobalArg, ImageArg

    from pymbolic import evaluate

    args = {}
    for arg, arg_desc in zip(kernel.args, arg_descriptors):
        if isinstance(arg, ValueArg):
            arg_value = parameters[arg.name]

            try:
                argv_dtype = arg_value.dtype
            except AttributeError:
                argv_dtype = None

            if argv_dtype != arg.dtype:
                arg_value = arg.dtype.type(arg_value)

            args[arg.name] = arg_value

        elif isinstance(arg, ImageArg):
            if arg.name in kernel.get_written_variables():
                raise NotImplementedError("write-mode images not supported in "
                        "automatic testing")

            shape = evaluate(arg.shape, parameters)
            assert shape == arg_desc.ref_shape

            # must be contiguous
            args[arg.name] = cl.image_from_array(
                    queue.context, arg_desc.ref_array.get())

        elif isinstance(arg, GlobalArg):
            assert arg.offset == 0

            shape = evaluate(arg.shape, parameters)
            strides = evaluate(arg.strides, parameters)

            itemsize = arg.dtype.itemsize
            numpy_strides = [itemsize*s for s in strides]

            assert all(s > 0 for s in strides)
            alloc_size = sum(astrd*(alen-1)
                    for alen, astrd in zip(shape, strides)) + 1

            if arg.name in kernel.get_written_variables():
                storage_array = cl_array.empty(queue, alloc_size, arg.dtype)
                ary = cl_array.as_strided(storage_array, shape, numpy_strides)

                if arg.dtype.isbuiltin:
                    storage_array.fill(fill_value)
                else:
                    from warnings import warn
                    warn("Cannot pre-fill array of dtype '%s'" % arg.dtype)

                args[arg.name] = ary
            else:
                # use contiguous array to transfer to host
                host_ref_contig_array = arg_desc.ref_storage_array.get()

                # use device shape/strides
                from pyopencl.compyte.array import as_strided
                host_ref_array = as_strided(host_ref_contig_array,
                        arg_desc.ref_shape, arg_desc.ref_numpy_strides)

                # flatten the thing
                host_ref_flat_array = host_ref_array.flatten()

                # create host array with test shape (but not strides)
                host_contig_array = np.empty(shape, dtype=arg.dtype)

                common_len = min(len(host_ref_flat_array), len(host_contig_array.ravel()))
                host_contig_array.ravel()[:common_len] = host_ref_flat_array[:common_len]

                # create host array with test shape and storage layout
                host_storage_array = np.empty(alloc_size, arg.dtype)
                host_array = as_strided(host_storage_array, shape, numpy_strides)
                host_array[:] = host_contig_array

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
            raise RuntimeError("arg type not understood")

    return args




def _default_check_result(result, ref_result):
    if not np.allclose(ref_result, result, rtol=1e-3, atol=1e-3):
        l2_err = np.sum(np.abs(ref_result-result)**2)/np.sum(np.abs(ref_result)**2)
        linf_err = np.max(np.abs(ref_result-result))/np.max(np.abs(ref_result-result))
        return (False,
                "results do not match(rel) l_2 err: %g, l_inf err: %g"
                % (l2_err, linf_err))
    else:
        return True, None




def _enumerate_cl_devices_for_ref_test():
    noncpu_devs = []
    cpu_devs = []

    from warnings import warn

    for pf in cl.get_platforms():
        if pf.name == "Portable OpenCL":
            # That implementation [1] isn't quite good enough yet.
            # [1] https://launchpad.net/pocl
            # FIXME remove when no longer true.
            warn("Skipping 'Portable OpenCL' for lack of maturity.")
            continue

        for dev in pf.get_devices():
            if dev.type == cl.device_type.CPU:
                cpu_devs.append(dev)
            else:
                noncpu_devs.append(dev)

    if not (cpu_devs or noncpu_devs):
        raise RuntimeError("no CL device found for test")

    if not cpu_devs:
        warn("No CPU device found for reference test. The reference computation "
                "will either fail because of a timeout or take a *very* long "
                "time.")

    for dev in cpu_devs:
        yield dev

    for dev in noncpu_devs:
        yield dev




def auto_test_vs_ref(ref_knl, ctx, kernel_gen, op_count=[], op_label=[], parameters={},
        print_ref_code=False, print_code=True, warmup_rounds=2,
        edit_code=False, dump_binary=False, codegen_kwargs={},
        options=[],
        fills_entire_output=True, do_check=True, check_result=None):
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

    read_and_written_args = (
            ref_knl.get_read_variables()
            & ref_knl.get_written_variables()
            & set(ref_knl.arg_dict))

    if read_and_written_args:
        # FIXME: In principle, that's possible to test
        raise RuntimeError("kernel reads *and* writes argument(s) '%s' "
                "and therefore cannot be automatically tested"
                % ", ".join(read_and_written_args))

    from time import time

    if check_result is None:
        check_result = _default_check_result

    if fills_entire_output:
        fill_value_ref = -17
        fill_value = -18
    else:
        fill_value_ref = -17
        fill_value = fill_value_ref

    # {{{ compile and run reference code

    found_ref_device = False

    ref_errors = []

    for dev in _enumerate_cl_devices_for_ref_test():
        ref_ctx = cl.Context([dev])
        ref_queue = cl.CommandQueue(ref_ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE)

        import loopy as lp
        ref_kernel_gen = lp.generate_loop_schedules(ref_knl)
        for knl in lp.check_kernels(ref_kernel_gen, parameters):
            ref_sched_kernel = knl
            break

        try:
            ref_args, arg_descriptors = \
                    make_ref_args(ref_sched_kernel, ref_queue, parameters,
                            fill_value=fill_value_ref)
            ref_args["out_host"] = False
        except cl.RuntimeError, e:
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

        ref_compiled = CompiledKernel(ref_ctx, ref_sched_kernel,
                options=options,
                codegen_kwargs=codegen_kwargs)
        if print_ref_code:
            print 75*"-"
            print "Reference Code:"
            print 75*"-"
            print get_highlighted_code(ref_compiled.code)
            print 75*"-"


        ref_queue.finish()
        ref_start = time()

        print "using %s for the reference calculation" % dev

        if not AUTO_TEST_SKIP_RUN:
            ref_evt, _ = ref_compiled(ref_queue, **ref_args)
        else:
            ref_evt = cl.enqueue_marker(ref_queue)

        ref_queue.finish()
        ref_stop = time()
        ref_elapsed_wall = ref_stop-ref_start

        ref_evt.wait()
        ref_elapsed = 1e-9*(ref_evt.profile.END-ref_evt.profile.SUBMIT)

        break

    if not found_ref_device:
        raise RuntimeError("could not find a suitable device for the reference computation.\n"
                "These errors were encountered:\n"+"\n".join(ref_errors))

    # }}}

    # {{{ compile and run parallel code

    need_check = do_check

    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    args = None
    for i, kernel in enumerate(kernel_gen):
        if args is None:
            args = make_args(queue, kernel, arg_descriptors, parameters,
                    fill_value=fill_value)
        args["out_host"] = False

        compiled = CompiledKernel(ctx, kernel, edit_code=edit_code,
                options=options,
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

        for i in range(warmup_rounds):
            if not AUTO_TEST_SKIP_RUN:
                compiled(queue, **args)

            if need_check and not AUTO_TEST_SKIP_RUN:
                for arg_desc in arg_descriptors:
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
                    assert error_is_small, error
                    need_check = False

        events = []
        queue.finish()

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
            rates += " %g %s/s" % (cnt/elapsed_wall, lbl)

        print "elapsed: %g s event, %s s marker-event %g s wall (%d rounds)%s" % (
                elapsed, elapsed_evt_2, elapsed_wall, timing_rounds, rates)

        if do_check:
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

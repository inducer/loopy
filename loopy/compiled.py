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


# {{{ domain parameter finder

class DomainParameterFinder(object):
    """Finds parameters from shapes of passed arguments."""

    def __init__(self, kernel, cl_arg_info):
        # a mapping from parameter names to a list of tuples
        # (arg_name, axis_nr, function), where function is a
        # unary function of kernel.arg_dict[arg_name].shape[axis_nr]
        # returning the desired parameter.
        self.param_to_sources = param_to_sources = {}

        param_names = kernel.all_params()

        from loopy.kernel.data import GlobalArg
        from loopy.symbolic import DependencyMapper
        from pymbolic import compile
        dep_map = DependencyMapper()

        from pymbolic import var
        for arg in cl_arg_info:
            if arg.arg_class is GlobalArg:
                for axis_nr, shape_i in enumerate(arg.shape):
                    deps = dep_map(shape_i)
                    if len(deps) == 1:
                        dep, = deps

                        if dep.name in param_names:
                            from pymbolic.algorithm import solve_affine_equations_for
                            try:
                                # friggin' overkill :)
                                param_expr = solve_affine_equations_for(
                                        [dep.name], [(shape_i, var("shape_i"))]
                                        )[dep.name]
                            except:
                                # went wrong? oh well
                                pass
                            else:
                                param_func = compile(param_expr, ["shape_i"])
                                param_to_sources.setdefault(dep.name, []).append(
                                        (arg.name, axis_nr, param_func))

    def __call__(self, kwargs):
        result = {}

        for param_name, sources in self.param_to_sources.iteritems():
            if param_name not in kwargs:
                for arg_name, axis_nr, shape_func in sources:
                    if arg_name in kwargs:
                        try:
                            shape_axis = kwargs[arg_name].shape[axis_nr]
                        except IndexError:
                            raise RuntimeError("Argument '%s' has unexpected shape. "
                                    "Tried to access axis %d (0-based), only %d "
                                    "axes present." %
                                    (arg_name, axis_nr, len(kwargs[arg_name].shape)))

                        result[param_name] = shape_func(shape_axis)
                        continue

        return result

# }}}


# {{{ argument checking

def _arg_matches_spec(arg, val, other_args):
    import loopy as lp
    if isinstance(arg, lp.GlobalArg):
        from pymbolic import evaluate

        if arg.dtype != val.dtype:
            raise TypeError("dtype mismatch on argument '%s' "
                    "(got: %s, expected: %s)"
                    % (arg.name, val.dtype, arg.dtype))

        if arg.shape is not None:
            shape = evaluate(arg.shape, other_args)
            if shape != val.shape:
                raise TypeError("shape mismatch on argument '%s' "
                        "(got: %s, expected: %s)"
                        % (arg.name, val.shape, shape))

        strides = evaluate(arg.numpy_strides, other_args)
        if strides != tuple(val.strides):
            raise ValueError("strides mismatch on argument '%s' "
                    "(got: %s, expected: %s)"
                    % (arg.name, val.strides, strides))

        if val.offset != 0 and arg.offset == 0:
            raise ValueError("Argument '%s' does not allow arrays "
                    "with offsets. Try passing default_offset=loopy.auto "
                    "to make_kernel()." % arg.name)

    return True

# }}}


# {{{ compiled kernel object

def _get_kernel_from_iterable(iterable):
    kernel_count = 0

    for scheduled_kernel in iterable:
        kernel_count += 1

        if kernel_count == 1:
            # use the first schedule
            result = scheduled_kernel

        if kernel_count == 2:
            from warnings import warn
            warn("kernel scheduling was ambiguous--more than one "
                    "schedule found, ignoring", stacklevel=2)
            break

    return result


class _KernelInfo(Record):
    pass


class CompiledKernel:
    def __init__(self, context, kernel, options=[], codegen_kwargs={}):
        """
        :arg kernel: may be a loopy.LoopKernel, a generator returning kernels
          (a warning will be issued if more than one is returned). If the
          kernel has not yet been loop-scheduled, that is done, too, with no
          specific arguments.
        """

        import loopy as lp

        # {{{ do scheduling, if not yet done

        if not isinstance(kernel, lp.LoopKernel):
            # someone threw us an iterable of kernels

            kernel = _get_kernel_from_iterable(kernel)

        # Whether we need to call check_kernels. Since we don't have parameter
        # values now, we'll do that on first invocation.

        # }}}

        self.context = context
        self.kernel = kernel
        self.codegen_kwargs = codegen_kwargs
        self.options = options

    @memoize_method
    def get_kernel_info(self, arg_to_dtype_set):
        kernel = self.kernel

        import loopy as lp
        from loopy.kernel.tools import add_argument_dtypes

        if arg_to_dtype_set:
            kernel = add_argument_dtypes(kernel, dict(arg_to_dtype_set))

            from loopy.preprocess import infer_unknown_types
            kernel = infer_unknown_types(kernel, expect_completion=True)

        if kernel.schedule is None:
            kernel = _get_kernel_from_iterable(
                    lp.generate_loop_schedules(kernel))

        # {{{ precompile, store grid size functions

        gsize_expr, lsize_expr = kernel.get_grid_sizes_as_exprs()

        if not gsize_expr:
            gsize_expr = (1,)
        if not lsize_expr:
            lsize_expr = (1,)

        # }}}

        from pymbolic import compile
        return _KernelInfo(
                kernel=kernel,
                global_size_func=compile(gsize_expr, kernel.scalar_loop_args),
                local_size_func=compile(lsize_expr, kernel.scalar_loop_args),
                )

    @memoize_method
    def cl_kernel_info(self,
            arg_to_dtype_set, code_op=False):
        kernel_info = self.get_kernel_info(arg_to_dtype_set)
        kernel = kernel_info.kernel

        from loopy.codegen import generate_code
        code, cl_arg_info = generate_code(kernel, **self.codegen_kwargs)

        if code_op == "print":
            print code
        elif code_op == "print_hl":
            print get_highlighted_code(code)
        elif code_op == "edit":
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
            print "[Loopy] "+70*"-"
            print "[Loopy] build failed, here's the source code:"
            print "[Loopy] "+70*"-"
            print code
            print "[Loopy] "+70*"-"
            print "[Loopy] end source code"
            print "[Loopy] "+70*"-"
            raise

        arg_types = []
        for arg_info in cl_arg_info:
            if arg_info.shape is None:
                arg_types.append(arg_info.dtype)
            else:
                arg_types.append(None)

        cl_kernel.set_scalar_arg_dtypes(arg_types)

        return kernel_info.copy(
                cl_kernel=cl_kernel,
                cl_arg_info=cl_arg_info,
                domain_parameter_finder=DomainParameterFinder(kernel, cl_arg_info))

    # {{{ debugging aids

    def get_code(self, arg_to_dtype=None):
        if arg_to_dtype is not None:
            arg_to_dtype = frozenset(arg_to_dtype.iteritems())

        kernel_info = self.get_kernel_info(arg_to_dtype)

        from loopy.codegen import generate_code
        code, arg_info = generate_code(kernel_info.kernel, **self.codegen_kwargs)
        return code

    def get_highlighted_code(self, arg_to_dtype=None):
        return get_highlighted_code(
                self.get_code(arg_to_dtype))

    @property
    def code(self):
        from warnings import warn
        warn("CompiledKernel.code is deprecated. Use .get_code() instead.",
                DeprecationWarning, stacklevel=2)

        return self.get_code()

    # }}}

    def __call__(self, queue, **kwargs):
        """If all array arguments are :mod:`numpy` arrays, defaults to
        returning numpy arrays as well.

        If you want offset arguments (see
        :attr:`loopy.kernel.data.GlobalArg.offset`) to be set automatically, it
        must occur *after* the corresponding array argument.
        """

        allocator = kwargs.pop("allocator", None)
        wait_for = kwargs.pop("wait_for", None)
        out_host = kwargs.pop("out_host", None)
        no_run = kwargs.pop("no_run", None)
        code_op = kwargs.pop("code_op", None)
        warn_numpy = kwargs.pop("warn_numpy", None)

        # {{{ process arg types, get cl kernel

        import loopy as lp

        arg_to_dtype = {}
        for arg in self.kernel.args:
            val = kwargs.get(arg.name)

            if arg.dtype is None and val is not None:
                try:
                    dtype = val.dtype
                except AttributeError:
                    pass
                else:
                    arg_to_dtype[arg.name] = dtype

        kernel_info = self.cl_kernel_info(
                frozenset(arg_to_dtype.iteritems()),
                code_op)
        kernel = kernel_info.kernel
        cl_kernel = kernel_info.cl_kernel
        del arg_to_dtype

        # }}}

        kwargs.update(
                kernel_info.domain_parameter_finder(kwargs))

        domain_parameters = dict((name, int(kwargs[name]))
                for name in kernel.scalar_loop_args)

        args = []
        outputs = []
        encountered_numpy = False
        encountered_cl = False

        kwargs_copy = kwargs.copy()

        for arg in kernel_info.cl_arg_info:
            is_written = arg.base_name in kernel.get_written_variables()

            val = kwargs_copy.pop(arg.name, None)

            # {{{ if this argument is an offset for another, try to determine it

            if arg.offset_for_name is not None and val is None:
                try:
                    array_arg_val = kwargs[arg.offset_for_name]
                except KeyError:
                    # Output variable, we'll be allocating it, with zero offset.
                    offset = 0
                else:
                    try:
                        offset = array_arg_val.offset
                    except AttributeError:
                        offset = 0

                if offset:
                    val, remdr = divmod(offset, array_arg_val.dtype.itemsize)
                    assert remdr == 0
                    del remdr
                else:
                    val = 0

                del offset

            # }}}

            if arg.shape is not None:
                # {{{ automatically transfer host-side arrays, if needed

                if isinstance(val, np.ndarray):
                    # synchronous, so nothing to worry about
                    val = cl_array.to_device(queue, val, allocator=allocator)
                    encountered_numpy = True
                    if warn_numpy:
                        from warnings import warn
                        warn("argument '%s' was passed as a numpy array, "
                                "performing implicit transfer" % arg.name,
                                stacklevel=2)
                else:
                    encountered_cl = True

                # }}}

            if val is None:
                if not is_written:
                    raise TypeError(
                            "must supply input argument '%s'" % arg.name)

                if arg.arg_class is lp.ImageArg:
                    raise RuntimeError("write-mode image '%s' must "
                            "be explicitly supplied" % arg.name)

                from pymbolic import evaluate
                shape = evaluate(arg.shape, kwargs)
                itemsize = arg.dtype.itemsize
                numpy_strides = tuple(
                        i*itemsize for i in evaluate(arg.strides, kwargs))

                from pytools import all
                assert all(s > 0 for s in numpy_strides)
                alloc_size = (sum(astrd*(alen-1)
                        for alen, astrd in zip(shape, numpy_strides))
                        + itemsize)

                if allocator is None:
                    storage = cl.Buffer(
                            queue.context, cl.mem_flags.READ_WRITE, alloc_size)
                else:
                    storage = allocator(alloc_size)

                val = cl_array.Array(queue, shape, arg.dtype,
                        strides=numpy_strides, data=storage,
                        allocator=allocator)
            else:
                assert _arg_matches_spec(arg, val, kwargs)

            if is_written:
                outputs.append(val)

            if arg.arg_class in [lp.GlobalArg, lp.ConstantArg]:
                args.append(val.base_data)
            else:
                args.append(val)

        assert not kwargs_copy, (
                "extra arguments: "+", ".join(kwargs_copy.iterkeys()))

        if no_run:
            evt = cl.enqueue_marker(queue)
        else:
            evt = cl_kernel(queue,
                    kernel_info.global_size_func(**domain_parameters),
                    kernel_info.local_size_func(**domain_parameters),
                    *args,
                    g_times_l=True, wait_for=wait_for)

        if out_host is None and (encountered_numpy and not encountered_cl):
            out_host = True
        if out_host:
            outputs = [o.get(queue=queue) for o in outputs]

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


# vim: foldmethod=marker

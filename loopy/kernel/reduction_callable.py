# Note: this file is just for convenience purposes. This would go back into
# kernel/function_interface.py.
# keeping it over here until everythin starts working.


from __future__ import division, absolute_import

from loopy.diagnostic import LoopyError

from loopy.kernel.function_interface import (InKernelCallable,
        ValueArgDescriptor)


class CallableReduction(InKernelCallable):

    fields = set(["operation", "arg_id_to_dtype", "arg_id_to_descr"])
    init_arg_names = ("operation", "arg_id_to_dtype", "arg_id_to_descr")

    def __init__(self, operation, arg_id_to_dtype=None,
            arg_id_to_descr=None, name_in_target=None):

        if isinstance(operation, str):
            from loopy.library.reduction import parse_reduction_op
            operation = parse_reduction_op(operation)

        from loopy.library.reduction import ReductionOperation
        assert isinstance(operation, ReductionOperation)

        self.operation = operation

        super(InKernelCallable, self).__init__(name="",
                arg_id_to_dtype=arg_id_to_dtype,
                arg_id_to_descr=arg_id_to_descr)

    def __getinitargs__(self):
        return (self.operation, self.arg_id_to_dtype,
                self.arg_id_to_descr)

    @property
    def is_tuple_typed(self):
        return self.operation.arg_count > 1

    def with_types(self, arg_id_to_dtype, target):
        if self.arg_id_to_dtype is not None:

            # specializing an already specialized function.

            for id, dtype in arg_id_to_dtype.items():
                # only checking for the ones which have been provided
                if self.arg_id_to_dtype[id] != arg_id_to_dtype[id]:
                    raise LoopyError("Overwriting a specialized"
                            " function is illegal--maybe start with new instance of"
                            " CallableScalar?")

        if self.name in target.get_device_ast_builder().function_identifiers():
            new_in_knl_callable = target.get_device_ast_builder().with_types(
                    self, arg_id_to_dtype)
            if new_in_knl_callable is None:
                new_in_knl_callable = self.copy()
            return new_in_knl_callable

        # did not find a scalar function and function prototype does not
        # even have  subkernel registered => no match found
        raise LoopyError("Function %s not present within"
                " the %s namespace" % (self.name, target))

    def with_descrs(self, arg_id_to_descr):

        # This is a scalar call
        # need to assert that the name is in funtion indentifiers
        arg_id_to_descr[-1] = ValueArgDescriptor()
        return self.copy(arg_id_to_descr=arg_id_to_descr)

    def with_iname_tag_usage(self, unusable, concurrent_shape):

        raise NotImplementedError()

    def is_ready_for_code_gen(self):

        return (self.arg_id_to_dtype is not None and
                self.arg_id_to_descr is not None and
                self.name_in_target is not None)


# vim: foldmethod=marker

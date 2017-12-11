import loopy as lp


# {{{ test_preamble_with_separate_temporaries

class SeparateTemporariesPreambleTestHelper:
    def __init__(self, func_name, func_arg_dtypes, func_result_dtypes, arr):
        self.func_name = func_name
        self.func_arg_dtypes = func_arg_dtypes
        self.func_result_dtypes = func_result_dtypes
        self.arr = arr

    def mangler(self, kernel, name, arg_dtypes):
        """
        A function that will return a :class:`loopy.kernel.data.CallMangleInfo`
        to interface with the calling :class:`loopy.LoopKernel`
        """
        if name != self.func_name:
            return None

        from loopy.types import to_loopy_type
        from loopy.kernel.data import CallMangleInfo

        def __compare(d1, d2):
            # compare dtypes ignoring atomic
            return to_loopy_type(d1, for_atomic=True) == \
                to_loopy_type(d2, for_atomic=True)

        # check types
        if len(arg_dtypes) != len(arg_dtypes):
            raise Exception('Unexpected number of arguments provided to mangler '
                            '{}, expected {}, got {}'.format(
                                self.func_name, len(self.func_arg_dtypes),
                                len(arg_dtypes)))

        for i, (d1, d2) in enumerate(zip(self.func_arg_dtypes, arg_dtypes)):
            if not __compare(d1, d2):
                raise Exception('Argument at index {} for mangler {} does not '
                                'match expected dtype.  Expected {}, got {}'.
                                format(i, self.func_name, str(d1), str(d2)))

        # get target for creation
        target = arg_dtypes[0].target
        return CallMangleInfo(
            target_name=self.func_name,
            result_dtypes=tuple(to_loopy_type(x, target=target) for x in
                                self.func_result_dtypes),
            arg_dtypes=arg_dtypes)

    def preamble_gen(self, preamble_info):
        from loopy.kernel.data import temp_var_scope as scopes

        # find a function matching our name
        func_match = next(
            (x for x in preamble_info.seen_functions
             if x.name == self.func_name), None)
        desc = 'custom_funcs_indirect'
        if func_match is not None:
            from loopy.types import to_loopy_type
            # check types
            if tuple(to_loopy_type(x) for x in self.func_arg_dtypes) == \
                    func_match.arg_dtypes:
                # if match, create our temporary
                var = lp.TemporaryVariable(
                    'lookup', initializer=self.arr, dtype=self.arr.dtype,
                    shape=self.arr.shape,
                    scope=scopes.GLOBAL, read_only=True)
                # and code
                code = """
        int {name}(int start, int end, int match)
        {{
            int result = start;
            for (int i = start + 1; i < end; ++i)
            {{
                if (lookup[i] == match)
                    result = i;
            }}
            return result;
        }}
        """.format(name=self.func_name)

        # generate temporary variable code
        from cgen import Initializer
        from loopy.target.c import generate_array_literal
        codegen_state = preamble_info.codegen_state.copy(
            is_generating_device_code=True)
        kernel = preamble_info.kernel
        ast_builder = codegen_state.ast_builder
        target = kernel.target
        decl_info, = var.decl_info(target, index_dtype=kernel.index_dtype)
        decl = ast_builder.wrap_global_constant(
                ast_builder.get_temporary_decl(
                    codegen_state, None, var,
                    decl_info))
        if var.initializer is not None:
            decl = Initializer(decl, generate_array_literal(
                codegen_state, var, var.initializer))
        # return generated code
        yield (desc, '\n'.join([str(decl), code]))

# }}}

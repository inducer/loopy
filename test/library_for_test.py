# This exists because function handles can't be pickled.


def no_ret_f_mangler(kernel, name, arg_dtypes):
    if not isinstance(name, str):
        return None

    if (name == "f" and len(arg_dtypes) == 0):
        from loopy.kernel.data import CallMangleInfo
        return CallMangleInfo(
                target_name="f",
                result_dtypes=arg_dtypes,
                arg_dtypes=arg_dtypes)


def no_ret_f_preamble_gen(preamble_info):
    yield ("10_define_f",
            r"""
            void f()
            {
                printf("Hi!\n");
            }
            """)

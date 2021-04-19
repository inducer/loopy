import loopy as lp


class NoRetFunction(lp.ScalarCallable):
    def with_types(self, arg_id_to_dtype, callables):
        if len(arg_id_to_dtype) != 0:
            raise RuntimeError("'f' cannot take any inputs.")

        return (self.copy(arg_id_to_dtype=arg_id_to_dtype,
                         name_in_target="f"),
                callables)

    def with_descrs(self, arg_id_to_descr, callables):
        if len(arg_id_to_descr) != 0:
            raise RuntimeError("'f' cannot take any inputs.")

        return (self.copy(arg_id_to_descr=arg_id_to_descr),
                callables)

    def generate_preambles(self, target):
        assert isinstance(target, lp.CFamilyTarget)
        yield ("10_define_f",
                r"""
                void f()
                {
                    printf("Hi!\n");
                }
                """)

import loopy as lp
import numpy as np


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


class SingleArgNoRetFunction(lp.ScalarCallable):
    def with_types(self, arg_id_to_dtype, callables):
        input_dtype = arg_id_to_dtype.get(0)
        if input_dtype is None:
            return self, callables

        if input_dtype.numpy_dtype != np.float32:
            raise RuntimeError("'f' only supports f32.")

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
                void f(float x)
                {
                    printf("Hi!\n");
                }
                """)


def symbol_x(knl, name):
    if name == "X":
        from loopy.types import to_loopy_type
        return to_loopy_type(np.float32), "X"


def preamble_for_x(preamble_info):
    yield ("preamble_ten", r"#define X 10.0")

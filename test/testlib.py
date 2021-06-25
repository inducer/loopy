import loopy as lp
import numpy as np


# {{{ test_barrier_in_overridden_get_grid_size_expanded_kernel

class GridOverride:
    def __init__(self, clean, vecsize):
        self.clean = clean
        self.vecsize = vecsize

    def __call__(self, insn_ids, callables_table, ignore_auto=True):
        gsize, _ = self.clean.get_grid_sizes_for_insn_ids(insn_ids,
                callables_table, ignore_auto)
        return gsize, (self.vecsize,)

# }}}


# {{{ test_register_function_lookup

class Log2Callable(lp.ScalarCallable):

    def with_types(self, arg_id_to_dtype, callables_table):

        if 0 not in arg_id_to_dtype or arg_id_to_dtype[0] is None:
            # the types provided aren't mature enough to specialize the
            # callable
            return (
                    self.copy(arg_id_to_dtype=arg_id_to_dtype),
                    callables_table)

        dtype = arg_id_to_dtype[0].numpy_dtype

        if dtype.kind in ("u", "i"):
            # ints and unsigned casted to float32
            dtype = np.float32

        if dtype.type == np.float32:
            name_in_target = "log2f"
        elif dtype.type == np.float64:
            name_in_target = "log2"
            pass
        else:
            raise TypeError(f"log2: unexpected type {dtype}")

        from loopy.types import NumpyType
        return (
                self.copy(name_in_target=name_in_target,
                    arg_id_to_dtype={0: NumpyType(dtype), -1:
                        NumpyType(dtype)}),
                callables_table)


# }}}

# vim: foldmethod=marker

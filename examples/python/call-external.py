import loopy as lp
import numpy as np
from loopy.diagnostic import LoopyError
from loopy.target.c import CTarget


# {{{ blas callable

class BLASCallable(lp.ScalarCallable):
    def with_types(self, arg_id_to_dtype, kernel, callables_table):
        for i in range(0, 2):
            if i not in arg_id_to_dtype or arg_id_to_dtype[i] is None:
                # the types provided aren't mature enough to specialize the
                # callable
                return (
                        self.copy(arg_id_to_dtype=arg_id_to_dtype),
                        callables_table)

        mat_dtype = arg_id_to_dtype[0].numpy_dtype
        vec_dtype = arg_id_to_dtype[1].numpy_dtype

        if mat_dtype != vec_dtype:
            raise LoopyError("DGEMV should have same dtype for matrix and "
                    "vector")

        if vec_dtype == np.float32:
            name_in_target = "cblas_sgemv"
        elif vec_dtype == np.float64:
            name_in_target = "cblas_dgemv"
        else:
            raise LoopyError("GEMV only supported for float32 and float64 "
                    "types")

        from loopy.types import NumpyType
        return self.copy(name_in_target=name_in_target,
                arg_id_to_dtype={0: NumpyType(vec_dtype), 1: NumpyType(vec_dtype),
                    -1: NumpyType(vec_dtype)}), callables_table

    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        assert self.is_ready_for_codegen()

        from loopy.kernel.instruction import CallInstruction

        assert isinstance(insn, CallInstruction)

        parameters = insn.expression.parameters

        parameters = list(parameters)
        par_dtypes = [self.arg_id_to_dtype[i] for i, _ in enumerate(parameters)]

        parameters.append(insn.assignees[0])
        par_dtypes.append(self.arg_id_to_dtype[-1])

        # no type casting in array calls.
        from loopy.expression import dtype_to_type_context
        from pymbolic.mapper.stringifier import PREC_NONE
        from loopy.symbolic import SubArrayRef
        from pymbolic import var

        mat_descr = self.arg_id_to_descr[0]

        c_parameters = [
                expression_to_code_mapper(par, PREC_NONE,
                    dtype_to_type_context(target, par_dtype),
                    par_dtype).expr if isinstance(par, SubArrayRef) else
                expression_to_code_mapper(par, PREC_NONE,
                    dtype_to_type_context(target, par_dtype),
                    par_dtype).expr
                for par, par_dtype in zip(
                    parameters, par_dtypes)]
        c_parameters.insert(0, var("CblasRowMajor"))
        c_parameters.insert(1, var("CblasNoTrans"))
        c_parameters.insert(2, mat_descr.shape[0])
        c_parameters.insert(3, mat_descr.shape[1])
        c_parameters.insert(4, 1)
        c_parameters.insert(6, 1)
        c_parameters.insert(8, 1)
        c_parameters.insert(10, 1)
        return var(self.name_in_target)(*c_parameters), False

    def generate_preambles(self, target):
        assert isinstance(target, CTarget)
        yield("99_cblas", "#include <cblas.h>")
        return


def blas_fn_lookup(target, identifier):
    if identifier == "gemv":
        return BLASCallable(name="gemv")
    return None

# }}}


n = 10

knl = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
        y[:] = gemv(A[:, :], x[:])
        """, [
            lp.GlobalArg("A", dtype=np.float64, shape=(n, n)),
            lp.GlobalArg("x", dtype=np.float64, shape=(n, )),
            lp.GlobalArg("y", shape=(n, )), ...],
        target=CTarget(),
        lang_version=(2018, 2))

knl = lp.register_function_id_to_in_knl_callable_mapper(
        knl, blas_fn_lookup)

print(lp.generate_code_v2(knl).device_code())

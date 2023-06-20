import loopy as lp
import numpy as np
from loopy.diagnostic import LoopyError
from loopy.target.c import CTarget
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


# {{{ blas callable

class CBLASGEMV(lp.ScalarCallable):
    def with_types(self, arg_id_to_dtype, callables_table):
        mat_dtype = arg_id_to_dtype.get(0)
        vec_dtype = arg_id_to_dtype.get(1)

        if mat_dtype is None or vec_dtype is None:
            # types aren't specialized enough to be resolved
            return self, callables_table

        if mat_dtype != vec_dtype:
            raise LoopyError("GEMV requires same dtypes for matrix and "
                             "vector")

        if vec_dtype.numpy_dtype == np.float32:
            name_in_target = "cblas_sgemv"
        elif vec_dtype. numpy_dtype == np.float64:
            name_in_target = "cblas_dgemv"
        else:
            raise LoopyError("GEMV is only supported for float32 and float64 "
                             "types")

        return (self.copy(name_in_target=name_in_target,
                          arg_id_to_dtype={0: vec_dtype,
                                           1: vec_dtype,
                                           -1: vec_dtype}),
                callables_table)

    def with_descrs(self, arg_id_to_descr, callables_table):
        mat_descr = arg_id_to_descr.get(0)
        vec_descr = arg_id_to_descr.get(1)
        res_descr = arg_id_to_descr.get(-1)

        if mat_descr is None or vec_descr is None or res_descr is None:
            # shapes aren't specialized enough to be resolved
            return self, callables_table

        assert mat_descr.shape[1] == vec_descr.shape[0]
        assert mat_descr.shape[0] == res_descr.shape[0]
        assert len(vec_descr.shape) == len(res_descr.shape) == 1
        # handling only the easy case when stride == 1
        assert vec_descr.dim_tags[0].stride == 1
        assert mat_descr.dim_tags[1].stride == 1
        assert res_descr.dim_tags[0].stride == 1

        return self.copy(arg_id_to_descr=arg_id_to_descr), callables_table

    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        from pymbolic import var
        mat_descr = self.arg_id_to_descr[0]
        m, n = mat_descr.shape
        ecm = expression_to_code_mapper
        mat, vec = insn.expression.parameters
        result, = insn.assignees

        c_parameters = [var("CblasRowMajor"),
                        var("CblasNoTrans"),
                        m, n,
                        1,
                        ecm(mat).expr,
                        1,
                        ecm(vec).expr,
                        1,
                        ecm(result).expr,
                        1]
        return (var(self.name_in_target)(*c_parameters),
                False  # cblas_gemv does not return anything
                )

    def generate_preambles(self, target):
        assert isinstance(target, CTarget)
        yield ("99_cblas", "#include <cblas.h>")
        return

# }}}


n = 10

knl = lp.make_kernel(
        "{:}",
        """
        y[:] = gemv(A[:, :], x[:])
        """, [
            lp.GlobalArg("A", dtype=np.float64, shape=(n, n)),
            lp.GlobalArg("x", dtype=np.float64, shape=(n, )),
            lp.GlobalArg("y", shape=(n, )), ...],
        target=CTarget())

knl = lp.register_callable(knl, "gemv", CBLASGEMV(name="gemv"))
print(lp.generate_code_v2(knl).device_code())

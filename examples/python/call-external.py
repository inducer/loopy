import loopy as lp
import numpy as np
from loopy.diagnostic import LoopyError
from loopy.target.c import CTarget
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401
from loopy.target.c.c_execution import CCompiler
from codepy.toolchain import GCCToolchain


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
        elif vec_dtype.numpy_dtype == np.float64:
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
        assert mat_descr.dim_tags[1].stride == 1

        return self.copy(arg_id_to_descr=arg_id_to_descr), callables_table

    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        from pymbolic import var
        from loopy.codegen import UnvectorizableError
        mat_descr = self.arg_id_to_descr[0]
        vec_descr = self.arg_id_to_descr[1]
        res_descr = self.arg_id_to_descr[-1]
        m, n = mat_descr.shape
        ecm = expression_to_code_mapper

        if ecm.codegen_state.vectorization_info is not None:
            raise UnvectorizableError("cannot vectorize BLAS-gemv.")

        mat, vec = insn.expression.parameters
        result, = insn.assignees

        c_parameters = [var("CblasRowMajor"),
                        var("CblasNoTrans"),
                        m, n,
                        1,  # alpha
                        ecm(mat).expr,
                        mat_descr.dim_tags[0].stride,  # LDA
                        ecm(vec).expr,
                        vec_descr.dim_tags[0].stride,  # INCX
                        0,  # beta
                        ecm(result).expr,
                        res_descr.dim_tags[0].stride  # INCY
                        ]
        return (var(self.name_in_target)(*c_parameters),
                False  # cblas_gemv does not return anything
                )

    def generate_preambles(self, target):
        assert isinstance(target, CTarget)
        yield("99_cblas", "#include <cblas.h>")
        return

# }}}


def transform_1(knl):
    return knl


def transform_2(knl):
    # A similar transformation is applied to kernels containing
    # SLATE <https://www.firedrakeproject.org/firedrake.slate.html>
    # callables.
    knl = lp.split_iname(knl, "e", 4, inner_iname="e_inner", slabs=(0, 1))
    knl = lp.privatize_temporaries_with_inames(knl, "e_inner")
    knl = lp.tag_inames(knl, {"e_inner": "vec"})
    if 0:
        # Easy codegen exercise, but misses vectorizing certain instructions.
        knl = lp.tag_array_axes(knl, "tmp3", "c,vec")
    else:
        knl = lp.tag_array_axes(knl, "tmp3,tmp2", "c,vec")
    return knl


def main():

    compiler = CCompiler(toolchain=GCCToolchain(
        cc="gcc",
        cflags="-std=c99 -O3 -fPIC".split(),
        ldflags="-shared".split(),
        libraries=["blas"],
        library_dirs=[],
        defines=[],
        undefines=[],
        source_suffix="c",
        so_ext=".so",
        o_ext=".o",
        include_dirs=[]))

    knl = lp.make_kernel(
        "{[e,i1,i2]: 0<=e<n and 0<=i1,i2<4}",
        """
        for e
            for i1
                tmp1[i1] = 3*x[e, i1]
            end
            tmp2[:] = matvec(A[:, :], tmp1[:])
            for i2
                <> tmp3[i2] = 2 * tmp2[i2]
                out[e, i2] = tmp3[i2]
            end
        end
        """,
        kernel_data=[
            lp.TemporaryVariable("tmp1",
                                 shape=(4, ),
                                 dtype=None),
            lp.TemporaryVariable("tmp2",
                                 shape=(4, ),
                                 dtype=None),
            lp.GlobalArg("A",
                         shape=(4, 4),
                         dtype="float64"),
            lp.GlobalArg("x",
                         shape=lp.auto,
                         dtype="float64"),
            ...],
        target=lp.ExecutableCVectorExtensionsTarget(compiler=compiler),
        lang_version=(2018, 2))

    knl = lp.register_callable(knl, "matvec", CBLASGEMV("matvec"))

    for transform_func in [transform_1, transform_2]:
        knl = transform_func(knl)
        print("Generated code from '{transform_func.__name__} -----'")
        print(lp.generate_code_v2(knl).device_code())
        print(75 * "-")

        # {{ verify the result is correct.

        from numpy.random import default_rng

        rng = default_rng(seed=0)
        a = rng.random((4, 4))
        x = rng.random((100, 4))

        _, (out,) = knl(A=a, x=x)

        np.testing.assert_allclose(6*np.einsum("ij,ej->ei",
                                               a, x),
                                   out)

        # }}}


if __name__ == "__main__":
    main()

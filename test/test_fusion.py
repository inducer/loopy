__copyright__ = "Copyright (C) 2021 University of Illinois Board of Trustees"

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


import loopy as lp
import numpy as np
import pyopencl as cl

from pyopencl.tools import \
    pytest_generate_tests_for_pyopencl as pytest_generate_tests  # noqa


def test_two_kernel_fusion(ctx_factory):
    """
    A simple fusion test of two sets of instructions.
    """

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knla = lp.make_kernel(
        "{[i]: 0<=i<10}",
        """
            out[i] = i
        """
    )
    knlb = lp.make_kernel(
        "{[j]: 0<=j<10}",
        """
            out[j] = j+100
        """
    )
    knl = lp.fuse_kernels([knla, knlb], data_flow=[("out", 0, 1)])
    evt, (out,) = knl(queue)
    np.testing.assert_allclose(out.get(), np.arange(100, 110))


def test_write_block_matrix_fusion(ctx_factory):
    """
    A slightly more complicated fusion test, where all
    sub-kernels write into the same global matrix, but
    in well-defined separate blocks. This tests makes sure
    data flow specification is preserved during fusion for
    matrix-assembly-like programs.
    """

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    def init_global_mat_prg():
        return lp.make_kernel(
            [
                "{[idof]: 0 <= idof < n}",
                "{[jdof]: 0 <= jdof < m}"
            ],
            """
                result[idof, jdof]  = 0 {id=init}
            """,
            [
                lp.GlobalArg("result", None,
                    shape="n, m",
                    offset=lp.auto),
                lp.ValueArg("n, m", np.int32),
                "...",
            ],
            options=lp.Options(
                return_dict=True
            ),
            default_offset=lp.auto,
            name="init_a_global_matrix",
        )

    def write_into_mat_prg():
        return lp.make_kernel(
            [
                "{[idof]: 0 <= idof < ndofs}",
                "{[jdof]: 0 <= jdof < mdofs}"
            ],
            """
                result[offset_i + idof, offset_j + jdof] = mat[idof, jdof]
            """,
            [
                lp.GlobalArg("result", None,
                    shape="n, m",
                    offset=lp.auto),
                lp.ValueArg("n, m", np.int32),
                lp.GlobalArg("mat", None,
                    shape="ndofs, mdofs",
                    offset=lp.auto),
                lp.ValueArg("offset_i", np.int32),
                lp.ValueArg("offset_j", np.int32),
                "...",
            ],
            options=lp.Options(
                return_dict=True
            ),
            default_offset=lp.auto,
            name="write_into_global_matrix",
        )

    # Construct a 2x2 diagonal matrix with
    # random 5x5 blocks on the block-diagonal,
    # and zeros elsewhere
    n = 10
    block_n = 5
    mat1 = np.random.randn(block_n, block_n)
    mat2 = np.random.randn(block_n, block_n)
    answer = np.block([[mat1, np.zeros((block_n, block_n))],
                      [np.zeros((block_n, block_n)), mat2]])
    kwargs = {"n": n, "m": n}

    # Do some renaming of individual programs before fusion
    kernels = [init_global_mat_prg()]
    for idx, (offset, mat) in enumerate([(0, mat1), (block_n, mat2)]):
        knl = lp.rename_argument(write_into_mat_prg(), "mat", f"mat_{idx}")
        kwargs[f"mat_{idx}"] = mat

        for iname in knl.default_entrypoint.all_inames():
            knl = lp.rename_iname(knl, iname, f"{iname}_{idx}")

        knl = lp.rename_argument(knl, "ndofs", f"ndofs_{idx}")
        knl = lp.rename_argument(knl, "mdofs", f"mdofs_{idx}")
        kwargs[f"ndofs_{idx}"] = block_n
        kwargs[f"mdofs_{idx}"] = block_n

        knl = lp.rename_argument(knl, "offset_i", f"offset_i_{idx}")
        knl = lp.rename_argument(knl, "offset_j", f"offset_j_{idx}")
        kwargs[f"offset_i_{idx}"] = offset
        kwargs[f"offset_j_{idx}"] = offset

        kernels.append(knl)

    fused_knl = lp.fuse_kernels(
        kernels,
        data_flow=[("result", 0, 1), ("result", 1, 2)],
    )
    fused_knl = lp.add_nosync(
        fused_knl,
        "global",
        "writes:result",
        "writes:result",
        bidirectional=True,
        force=True
    )
    evt, result = fused_knl(queue, **kwargs)
    result = result["result"]
    np.testing.assert_allclose(result, answer)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

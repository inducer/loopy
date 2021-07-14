__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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


import sys
import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.clrandom  # noqa
import pytest

import logging
logger = logging.getLogger(__name__)

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests

__all__ = [
        "pytest_generate_tests",
        "cl"  # "cl.create_some_context"
        ]


pytest.importorskip("fparser")


def test_fp_prec_comparison():
    # FIXME: This test should succeed even when the number is exactly
    # representable in single precision.
    #
    # https://gitlab.tiker.net/inducer/loopy/issues/187

    fortran_src_dp = """
        subroutine assign_scalar(a)
          real*8 a(1)

          a(1) = 1.1d0
        end
        """

    prg_dp = lp.parse_fortran(fortran_src_dp)

    fortran_src_sp = """
        subroutine assign_scalar(a)
          real*8 a(1)

          a(1) = 1.1
        end
        """

    prg_sp = lp.parse_fortran(fortran_src_sp)

    assert prg_sp != prg_dp


def test_assign_double_precision_scalar(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    fortran_src = """
        subroutine assign_scalar(a)
          real*8 a(1)

          a(1) = 1.1d0
        end
        """

    t_unit = lp.parse_fortran(fortran_src)
    print(lp.generate_code_v2(t_unit).device_code())
    assert "1.1;" in lp.generate_code_v2(t_unit).device_code()

    a_dev = cl.array.empty(queue, 1, dtype=np.float64, order="F")
    t_unit(queue, a=a_dev)

    abs_err = abs(a_dev.get()[0] - 1.1)
    assert abs_err < 1e-15


def test_assign_double_precision_scalar_as_rational(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    fortran_src = """
        subroutine assign_scalar(a)
          real*8 a(1)

          a(1) = 11
          a(1) = a(1) / 10
        end
        """

    t_unit = lp.parse_fortran(fortran_src)

    a_dev = cl.array.empty(queue, 1, dtype=np.float64, order="F")
    t_unit(queue, a=a_dev)

    abs_err = abs(a_dev.get()[0] - 1.1)
    assert abs_err < 1e-15


def test_assign_single_precision_scalar(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    fortran_src = """
        subroutine assign_scalar(a)
          real*8 a(1)

          a(1) = 1.1
        end
        """

    t_unit = lp.parse_fortran(fortran_src)
    assert "1.1f" in lp.generate_code_v2(t_unit).device_code()

    a_dev = cl.array.empty(queue, 1, dtype=np.float64, order="F")
    t_unit(queue, a=a_dev)

    abs_err = abs(a_dev.get()[0] - 1.1)
    assert abs_err > 1e-15
    assert abs_err < 1e-6


def test_fill(ctx_factory):
    fortran_src = """
        subroutine fill(out, a, n)
          implicit none

          real*8 a, out(n)
          integer n, i

          do i = 1, n
            out(i) = a
          end do
        end

        !$loopy begin
        !
        ! fill = lp.parse_fortran(SOURCE)
        ! fill = lp.split_iname(fill, "i", split_amount,
        !     outer_tag="g.0", inner_tag="l.0")
        ! RESULT = fill
        !
        !$loopy end
        """

    knl = lp.parse_transformed_fortran(fortran_src,
            pre_transform_code="split_amount = 128")

    assert "i_inner" in knl["fill"].all_inames()

    ctx = ctx_factory()

    lp.auto_test_vs_ref(knl, ctx, knl, parameters=dict(n=5, a=5))


def test_fill_const(ctx_factory):
    fortran_src = """
        subroutine fill(out, a, n)
          implicit none

          real*8 a, out(n)
          integer n, i

          do i = 1, n
            out(i) = 3.45
          end do
        end
        """

    knl = lp.parse_fortran(fortran_src)

    ctx = ctx_factory()

    lp.auto_test_vs_ref(knl, ctx, knl, parameters=dict(n=5, a=5))


def test_asterisk_in_shape(ctx_factory):
    fortran_src = """
        subroutine fill(out, out2, inp, n)
          implicit none

          real*8 a, out(n), out2(n), inp(*)
          integer n, i

          do i = 1, n
            a = inp(n)
            out(i) = 5*a
            out2(i) = 6*a
          end do
        end
        """

    knl = lp.parse_fortran(fortran_src)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl(queue, inp=np.array([1, 2, 3.]), n=3)


def test_assignment_to_subst(ctx_factory):
    fortran_src = """
        subroutine fill(out, out2, inp, n)
          implicit none

          real*8 a, out(n), out2(n), inp(n)
          integer n, i

          do i = 1, n
            a = inp(i)
            out(i) = 5*a
            out2(i) = 6*a
          end do
        end
        """

    knl = lp.parse_fortran(fortran_src)

    ref_knl = knl

    knl = lp.assignment_to_subst(knl, "a", "i")

    ctx = ctx_factory()
    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=5))


def test_assignment_to_subst_two_defs(ctx_factory):
    fortran_src = """
        subroutine fill(out, out2, inp, n)
          implicit none

          real*8 a, out(n), out2(n), inp(n)
          integer n, i

          do i = 1, n
            a = inp(i)
            out(i) = 5*a
            a = 3*inp(n)
            out2(i) = 6*a
          end do
        end
        """

    knl = lp.parse_fortran(fortran_src)

    ref_knl = knl

    knl = lp.assignment_to_subst(knl, "a")

    ctx = ctx_factory()
    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=5))


def test_assignment_to_subst_indices(ctx_factory):
    fortran_src = """
        subroutine fill(out, out2, inp, n)
          implicit none

          real*8 a(n), out(n), out2(n), inp(n)
          integer n, i

          do i = 1, n
            a(i) = 6*inp(i)
          enddo

          do i = 1, n
            out(i) = 5*a(i)
          end do
        end
        """

    knl = lp.parse_fortran(fortran_src)

    knl = lp.fix_parameters(knl, n=5)

    ref_knl = knl

    assert "a" in knl["fill"].temporary_variables
    knl = lp.assignment_to_subst(knl, "a")
    assert "a" not in knl["fill"].temporary_variables

    ctx = ctx_factory()
    lp.auto_test_vs_ref(ref_knl, ctx, knl)


def test_if(ctx_factory):
    fortran_src = """
        subroutine fill(out, out2, inp, n)
          implicit none

          real*8 a, b, out(n), out2(n), inp(n)
          integer n, i, j

          do i = 1, n
            a = inp(i)
            if (a.ge.3) then
                b = 2*a
                do j = 1,3
                    b = 3 * b
                end do
                out(i) = 5*b
            else
                out(i) = 4*a
            endif
          end do
        end
        """

    knl = lp.parse_fortran(fortran_src)

    ref_knl = knl

    knl = lp.assignment_to_subst(knl, "a")

    ctx = ctx_factory()
    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=5))


def test_tagged(ctx_factory):
    fortran_src = """
        subroutine rot_norm(out, alpha, out2, inp, inp2, n)
          implicit none
          real*8 a, b, r, out(n), out2(n), inp(n), inp2(n)
          real*8 alpha
          integer n, i

          do i = 1, n
            !$loopy begin tagged: input
            a = cos(alpha)*inp(i) + sin(alpha)*inp2(i)
            b = -sin(alpha)*inp(i) + cos(alpha)*inp2(i)
            !$loopy end tagged: input

            r = sqrt(a**2 + b**2)
            a = a/r
            b = b/r

            out(i) = a
            out2(i) = b
          end do
        end
        """

    knl = lp.parse_fortran(fortran_src)

    assert sum(1 for insn in lp.find_instructions(knl, "tag:input")) == 2


@pytest.mark.parametrize("buffer_inames", [
    "",
    "i_inner,j_inner",
    ])
def test_matmul(ctx_factory, buffer_inames):
    ctx = ctx_factory()

    if (buffer_inames and
            ctx.devices[0].platform.name == "Portable Computing Language"):
        pytest.skip("crashes on pocl")

    logging.basicConfig(level=logging.INFO)

    fortran_src = """
        subroutine dgemm(m,n,ell,a,b,c)
          implicit none
          real*8 a(m,ell),b(ell,n),c(m,n)
          integer m,n,k,i,j,ell

          do j = 1,n
            do i = 1,m
              do k = 1,ell
                c(i,j) = c(i,j) + b(k,j)*a(i,k)
              end do
            end do
          end do
        end subroutine
        """

    prog = lp.parse_fortran(fortran_src)

    assert len(prog["dgemm"].domains) == 1

    ref_prog = prog

    prog = lp.split_iname(prog, "i", 16,
            outer_tag="g.0", inner_tag="l.1")
    prog = lp.split_iname(prog, "j", 8,
            outer_tag="g.1", inner_tag="l.0")
    prog = lp.split_iname(prog, "k", 32)
    prog = lp.assume(prog, "n mod 32 = 0")
    prog = lp.assume(prog, "m mod 32 = 0")
    prog = lp.assume(prog, "ell mod 16 = 0")

    prog = lp.extract_subst(prog, "a_acc", "a[i1,i2]", parameters="i1, i2")
    prog = lp.extract_subst(prog, "b_acc", "b[i1,i2]", parameters="i1, i2")
    prog = lp.precompute(prog, "a_acc", "k_inner,i_inner",
            precompute_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")
    prog = lp.precompute(prog, "b_acc", "j_inner,k_inner",
            precompute_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")

    prog = lp.buffer_array(prog, "c", buffer_inames=buffer_inames,
            init_expression="0", store_expression="base+buffer")

    lp.auto_test_vs_ref(ref_prog, ctx, prog, parameters=dict(n=128, m=128, ell=128))


@pytest.mark.xfail
def test_batched_sparse():
    fortran_src = """
        subroutine sparse(rowstarts, colindices, values, m, n, nvecs, nvals, x, y)
          implicit none

          integer rowstarts(m+1), colindices(nvals)
          real*8 values(nvals)
          real*8 x(n, nvecs), y(n, nvecs), rowsum(nvecs)

          integer m, n, rowstart, rowend, length, nvals, nvecs
          integer i, j, k

          do i = 1, m
            rowstart = rowstarts(i)
            rowend = rowstarts(i+1)
            length = rowend - rowstart

            do k = 1, nvecs
              rowsum(k) = 0
            enddo
            do k = 1, nvecs
              do j = 1, length
                rowsum(k) = rowsum(k) + &
                  x(colindices(rowstart+j-1),k)*values(rowstart+j-1)
              end do
            end do
            do k = 1, nvecs
              y(i,k) = rowsum(k)
            end do
          end do
        end

        """

    knl = lp.parse_fortran(fortran_src)

    knl = lp.split_iname(knl, "i", 128)
    knl = lp.tag_inames(knl, {"i_outer": "g.0"})
    knl = lp.tag_inames(knl, {"i_inner": "l.0"})
    knl = lp.add_prefetch(knl, "values",
            default_tag="l.auto")
    knl = lp.add_prefetch(knl, "colindices",
            default_tag="l.auto")
    knl = lp.fix_parameters(knl, nvecs=4)


def test_fuse_kernels(ctx_factory):
    fortran_template = """
        subroutine {name}(nelements, ndofs, result, d, q)
          implicit none
          integer e, i, j, k
          integer nelements, ndofs
          real*8 result(nelements, ndofs, ndofs)
          real*8 q(nelements, ndofs, ndofs)
          real*8 d(ndofs, ndofs)
          real*8 prev

          do e = 1,nelements
            do i = 1,ndofs
              do j = 1,ndofs
                do k = 1,ndofs
                  {inner}
                end do
              end do
            end do
          end do
        end subroutine
        """

    xd_line = """
        prev = result(e,i,j)
        result(e,i,j) = prev + d(i,k)*q(e,i,k)
        """
    yd_line = """
        prev = result(e,i,j)
        result(e,i,j) = prev + d(i,k)*q(e,k,j)
        """

    xderiv = lp.parse_fortran(
            fortran_template.format(inner=xd_line, name="xderiv"))
    yderiv = lp.parse_fortran(
            fortran_template.format(inner=yd_line, name="yderiv"))
    xyderiv = lp.parse_fortran(
            fortran_template.format(
                inner=(xd_line + "\n" + yd_line), name="xyderiv"))

    knl = lp.fuse_kernels((xderiv["xderiv"], yderiv["yderiv"]),
            data_flow=[("result", 0, 1)])
    knl = knl.with_kernel(lp.prioritize_loops(knl["xderiv_and_yderiv"], "e,i,j,k"))

    assert len(knl["xderiv_and_yderiv"].temporary_variables) == 2

    ctx = ctx_factory()
    lp.auto_test_vs_ref(xyderiv, ctx, knl, parameters=dict(nelements=20, ndofs=4))


def test_parse_and_fuse_two_kernels():
    fortran_src = """
        subroutine fill(out, a, n)
          implicit none

          real*8 a, out(n)
          integer n, i

          do i = 1, n
            out(i) = a
          end do
        end

        subroutine twice(out, n)
          implicit none

          real*8 out(n)
          integer n, i

          do i = 1, n
            out(i) = 2*out(i)
          end do
        end

        !$loopy begin
        !
        ! t_unit = lp.parse_fortran(SOURCE)
        ! fill = t_unit["fill"]
        ! twice = t_unit["twice"]
        ! knl = lp.fuse_kernels((fill, twice))
        ! print(knl)
        ! RESULT = knl
        !
        !$loopy end
        """

    lp.parse_transformed_fortran(fortran_src)


def test_precompute_some_exist(ctx_factory):
    fortran_src = """
        subroutine dgemm(m,n,ell,a,b,c)
          implicit none
          real*8 a(m,ell),b(ell,n),c(m,n)
          integer m,n,k,i,j,ell

          do j = 1,n
            do i = 1,m
              do k = 1,ell
                c(i,j) = c(i,j) + b(k,j)*a(i,k)
              end do
            end do
          end do
        end subroutine
        """

    knl = lp.parse_fortran(fortran_src)

    assert len(knl["dgemm"].domains) == 1

    knl = lp.split_iname(knl, "i", 8,
            outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_iname(knl, "j", 8,
            outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_iname(knl, "k", 8)
    knl = lp.assume(knl, "n mod 8 = 0")
    knl = lp.assume(knl, "m mod 8 = 0")
    knl = lp.assume(knl, "ell mod 8 = 0")

    knl = lp.extract_subst(knl, "a_acc", "a[i1,i2]", parameters="i1, i2")
    knl = lp.extract_subst(knl, "b_acc", "b[i1,i2]", parameters="i1, i2")
    knl = lp.precompute(knl, "a_acc", "k_inner,i_inner",
            precompute_inames="ktemp,itemp",
            precompute_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")
    knl = lp.precompute(knl, "b_acc", "j_inner,k_inner",
            precompute_inames="itemp,k2temp",
            precompute_outer_inames="i_outer, j_outer, k_outer",
            default_tag="l.auto")

    ref_knl = knl

    ctx = ctx_factory()
    lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=128, m=128, ell=128))


def test_fortran_subroutines():
    fortran_src = """
        subroutine twice(n, a)
          implicit none
          real*8  a(n)
          integer i,n

          do i=1,n
            a(i) = a(i) * 2
          end do
        end subroutine

        subroutine twice_cross(n, a, i)
          implicit none
          integer i, n
          real*8  a(n,n)

          call twice(n, a(1:n, i))
          call twice(n, a(i, 1:n))
        end subroutine
        """
    t_unit = lp.parse_fortran(fortran_src).with_entrypoints("twice_cross")
    print(lp.generate_code_v2(t_unit).device_code())


def test_domain_fusion_imperfectly_nested():
    fortran_src = """
        subroutine imperfect(n, m, a, b)
            implicit none
            integer i, j, n, m
            real a(n), b(n,n)

            do i=1, n
                a(i) = i
                do j=1, m
                    b(i,j) = i*j
                end do
            end do
        end subroutine
        """

    t_unit = lp.parse_fortran(fortran_src)
    # If n > 0 and m == 0, a single domain would be empty,
    # leading (incorrectly) to no assignments to 'a'.
    assert len(t_unit["imperfect"].domains) > 1


def test_division_in_shapes(ctx_factory):
    fortran_src = """
        subroutine halve(m, a)
            implicit none
            integer m, i, j
            real*8 a(m/2,m/2)
            do i = 1,m/2
                do j = 1,m/2
                    a(i, j) = 2*a(i, j)
                end do
            end do
        end subroutine
        """
    t_unit = lp.parse_fortran(fortran_src)
    ref_t_unit = t_unit

    print(t_unit)

    ctx = ctx_factory()
    lp.auto_test_vs_ref(ref_t_unit, ctx, t_unit, parameters=dict(m=128))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker

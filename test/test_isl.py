__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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

import islpy as isl


def test_aff_to_expr():
    s = isl.Space.create_from_names(isl.Context(), ["a", "b"])
    zero = isl.Aff.zero_on_domain(isl.LocalSpace.from_space(s))
    one = zero.set_constant_val(1)  # noqa
    a = zero.set_coefficient_val(isl.dim_type.in_, 0, 1)
    b = zero.set_coefficient_val(isl.dim_type.in_, 1, 1)

    x = (5*a + 3*b) % 17 % 5
    print(x)
    from loopy.symbolic import aff_to_expr
    print(aff_to_expr(x))


def test_aff_to_expr_2():
    from loopy.symbolic import aff_to_expr
    x = isl.Aff("[n] -> { [i0] -> [(-i0 + 2*floor((i0)/2))] }")
    from pymbolic import var
    i0 = var("i0")
    assert aff_to_expr(x) == (-1)*i0 + 2*(i0 // 2)


def test_pw_aff_to_conditional_expr():
    from loopy.symbolic import pw_aff_to_expr
    cond = isl.PwAff("[i] -> { [(0)] : i = 0; [(-1 + i)] : i > 0 }")
    expr = pw_aff_to_expr(cond)
    assert str(expr) == "0 if i == 0 else -1 + i"


def test_subst_into_pwqpolynomial():
    from pymbolic.primitives import Variable
    arg_dict = {
            "m": 3*Variable("nx"),
            "n": 3*Variable("ny"),
            "nx": Variable("nx"),
            "ny": Variable("ny"),
            "nz": Variable("nz")}
    space = isl.Set("[nx, ny, nz] -> { []: }").space
    poly = isl.PwQPolynomial("[m, n] -> { (256 * m + 256 * m * n) : "
        "m > 0 and n > 0; 256 * m : m > 0 and n <= 0 }")

    from loopy.isl_helpers import subst_into_pwqpolynomial
    result = subst_into_pwqpolynomial(space, poly, arg_dict)
    expected_pwqpoly = isl.PwQPolynomial("[nx, ny, nz] -> {"
            "(768 * nx + 2304 * nx * ny) : nx > 0 and ny > 0;"
            "768 * nx : nx > 0 and ny <= 0 }")
    assert (result - expected_pwqpoly).is_zero()


def test_subst_into_pwaff():
    from pymbolic.primitives import Variable
    arg_dict = {
            "m": 3*Variable("nx"),
            "n": 2*Variable("ny")+4}
    space = isl.Set("[nx, ny, nz] -> { []: }").params().space
    poly = isl.PwAff("[m, n] -> { [3 * m + 2 * n] : "
        "m > 0 and n > 0; [7* m + 4*n] : m > 0 and n <= 0 }")

    from loopy.isl_helpers import subst_into_pwaff
    result = subst_into_pwaff(space, poly, arg_dict)
    expected = isl.PwAff("[nx, ny, nz] -> { [(9nx + 4ny+8)] : nx > 0 and ny > -2;"
            " [(21nx + 8ny+16)] : nx > 0 and ny <= -2 }")
    assert result == expected


def test_simplify_via_aff_reproducibility():
    # See https://github.com/inducer/loopy/pull/349
    from loopy.symbolic import parse, simplify_via_aff

    expr = parse("i+i_0")

    assert simplify_via_aff(expr) == expr


def test_qpolynomrial_to_expr():
    from loopy.symbolic import qpolynomial_to_expr
    import pymbolic.primitives as p

    (_, qpoly), = isl.PwQPolynomial(
        "[i,j,k] -> { ((1/3)*i + (1/2)*j + (1/4)*k) : (4i+6j+3k) mod 12 = 0}"
    ).get_pieces()

    expr = qpolynomial_to_expr(qpoly)

    i, j, k = p.variables("i j k")
    assert expr == (4*i + 6*j + 3*k) // 12


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

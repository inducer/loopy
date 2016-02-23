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


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
